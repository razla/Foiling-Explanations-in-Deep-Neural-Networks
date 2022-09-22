
import sys
print(sys.prefix)
from copyreg import pickle
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import argparse
import os.path
import torch
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR, SequentialLR
from scipy.stats import qmc, norm

from nn.org_utils import plot_overview, clamp, load_image, make_dir

from utils import load_images, load_model, get_mean_std, label_to_name, get_optimizer, get_expl
from compression import Compression_3_channels
from stats import get_std_grad

import warnings
warnings.filterwarnings("ignore")

def get_beta(i, n_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / n_iter)

# def main():
argparser = argparse.ArgumentParser()
argparser.add_argument('--n_iter', type=int, default=2, help='number of iterations')
argparser.add_argument('--n_pop', type=int, default=2, choices=[50, 100, 200], help='number of individuals sampled from gaussian')
argparser.add_argument('--mean', type=float, default=0, help='mean of the gaussian distribution')
argparser.add_argument('--std', type=float, default=0.1, help='std of the gaussian distribution')
argparser.add_argument('--lr', type=float, default=0.0125,choices=[0.025, 0.0125, 0.00625], help='learning rate')
argparser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
argparser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10'], help='') #later 'cifar100' 'cifar10'
argparser.add_argument('--model', type=str, default='vgg16', help='model to use')
argparser.add_argument('--n_imgs', type=int, default=5, help='number of images to execute on')
argparser.add_argument('--img', type=str, default='../data/collie.jpeg', help='image net file to run attack on')
argparser.add_argument('--target_img', type=str, default='../data/tiger_cat.jpeg',
                        help='imagenet file used to generate target expl')
argparser.add_argument('--output_dir', type=str, default='output/', help='directory to save results to')
argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
argparser.add_argument('--is_scalar', help='is std a scalar', type=int, choices=[0,1], default=1) #later
argparser.add_argument('--to_compress', help='applying compression', type=int, choices=[0,1], default=0)
argparser.add_argument('--compression_method', help='PCA or SVD', type=str, default='PCA')
argparser.add_argument('--n_components', help='How many principle components',choices=[175], type=int, default=150)
argparser.add_argument('--latin_sampling', help='sample with latin hypercube', type=int, choices=[0,1], default=1)
argparser.add_argument('--synthesize', help='synthesizing target image to org image', type=int, choices=[0,1], default=0)
argparser.add_argument('--uniPixel', help='treating RGB values as one', type=int, choices=[0,1], default=0) #later
argparser.add_argument('--std_grad_update', help='using gradient update for the std', type=int, choices=[0,1], default=1)
argparser.add_argument('--std_exp_update', help='using exponential decay for the std', type=float, default=0.99) # later
argparser.add_argument('--MC_FGSM', help='using MC-FGSM gradient update', type=int, choices=[0,1], default=0) # later
argparser.add_argument('--max_delta', help='maximum change in image', type=float, default=1.0)
argparser.add_argument('--optimizer', help='', choices=['Adam', 'SGD', 'RMSprop'], type=str, default='Adam')
argparser.add_argument('--weight_decay', help='', choices=[0.0, 0.0001], type=float, default=0.0)

argparser.add_argument('--prefactors', nargs=4, default=[1e11, 1e6, 1e4, 1e2], type=float,
                        help='prefactors of losses (diff expls, class loss, l2 loss, l1 loss)')
argparser.add_argument('--method', help='algorithm for expls',
                        choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                'pattern_attribution', 'grad_times_input'],
                        default='integrated_grad')

args = argparser.parse_args()


print(f'args.to_compress: {args.to_compress}', flush=True)

n_iter = args.n_iter
n_pop = args.n_pop
uniPixel = args.uniPixel
max_delta = args.max_delta
is_scalar = args.is_scalar
w_decay = args.weight_decay
opt = args.optimizer
experiment = f'n_iter_{n_iter}_n_pop_{n_pop}_method_{args.method}_lr_{args.lr}_max_delta_{max_delta}_opt_{opt}_w_decay_{w_decay}'
if args.to_compress:
    if args.compression_method.lower() == 'pca':
        experiment+='_PCA'
    else:
        experiment+='_SVD'
    experiment += f'_{args.n_components}'
if args.latin_sampling:
    experiment += f'_LS'
if args.synthesize:
    experiment += f'_SYN'
if args.MC_FGSM:
    experiment += f'_MC_FGSM'
if args.uniPixel:
    experiment += f'_uniPixel'
if opt.lower() in ['sgd', 'rmsprop']:
    experiment += f'_mu_{args.momentum}'
if is_scalar:
    experiment += f'_scalar_std'
if args.std_grad_update:
    experiment += f'_std_grad_update'
else:
    experiment += f'_std_exp_update_{args.std_exp_update}'

experiment += f'_prefactors_{str(args.prefactors)}'


seed = 0
experiment += f'_seed_{seed}'
np.save('argparser/' + experiment + '.npy', args.__dict__, allow_pickle=True)
print(experiment, flush=True)

experiment = 'debug'
print(experiment)

# options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load model
data_mean, data_std = get_mean_std(args.dataset)
pretrained_model = load_model(args.model, args.dataset, device)
model = pretrained_model.eval().to(device)

base_images_paths, target_images_paths = load_images(args.n_imgs, args.dataset, seed)
for index, (base_image, target_image) in enumerate(zip(base_images_paths, target_images_paths)):
    loss_expl_list = []
    loss_input_list = []
    loss_output_list = []
    mean = torch.tensor(args.mean)
    std = torch.tensor(args.std)
    lr = args.lr
    mu = args.momentum
    subset_idx_threshold = 0
    # load images
    x = load_image(data_mean, data_std, device, base_image, args.dataset)
    x_target = load_image(data_mean, data_std, device, target_image, args.dataset)
    if x is None or x_target is None:
        continue
    if args.synthesize:
        x_adv = x_target.clone().detach().requires_grad_()
    else:
        x_adv = x.clone().detach().requires_grad_()
    x_noise = x.clone().detach().requires_grad_()
    # produce expls
    org_expl, org_acc, org_idx = get_expl(model, x, args.method)
    org_expl = org_expl.detach().cpu()
    org_label_name = label_to_name(org_idx.item(), args.dataset)
    target_expl, _, target_idx = get_expl(model, x_target, args.method)
    target_expl = target_expl.detach()
    target_label_name = label_to_name(target_idx.item(), args.dataset)

    print(org_label_name, base_image)
    print(target_label_name, target_image)

    total_loss_list = torch.Tensor(n_pop).to(device)
    best_X_adv = deepcopy(x_adv)
    best_loss = float('inf')

    if args.to_compress:
        pca_3 = Compression_3_channels(args.n_components, method=args.compression_method)
        x_compressed = pca_3.fit_transform(x.detach().cpu().numpy()[0].T)
        img_recon = pca_3.inverse_transform()
        x_recon = torch.tensor(img_recon.T).unsqueeze(0).to(device)
        x_compressed = torch.tensor(x_compressed.T).unsqueeze(0).to(device)
        x_adv_comp = x_compressed.clone()
        x_noise = x_compressed.clone()

    if uniPixel:
        x_noise = x_noise[:,0:1,:,:]
        V = x_noise.clone().detach().zero_()
    noise_list = [x_noise.clone().detach().data.normal_(mean.item(), std.item()).requires_grad_() for _ in range(n_pop)]
    noise_list[0] = x_noise.clone().detach().zero_().requires_grad_()
    if args.latin_sampling:
        sampler = qmc.LatinHypercube(d=np.product(x_noise.shape), optimization=None) # optimization = None is faster, "random-cd" , strength=1, centered=False
        sample = torch.tensor(norm(loc=mean.item(), scale=std.item()).ppf(sampler.random(n=n_pop-1))).to(device)
        for k in range(1, len(noise_list)):
            noise_list[k].data = sample[k-1].reshape(x_noise.shape)
    V = x_noise.clone().detach().zero_()

    optimizer = get_optimizer(opt, V, lr, mu, w_decay)

    scheduler = ExponentialLR(optimizer, gamma=0.995)

    for i in range(args.n_iter):

        loss_expl_0 = None
        loss_output_0 = None
        for j, noise in enumerate(noise_list):
            if args.to_compress:
                x_adv_recon = pca_3.inverse_transform_noise(V.data.T.cpu().numpy() + noise.data.T.cpu().numpy(), uniPixel) # 3 layer update
                x_adv_recon = torch.tensor(x_adv_recon.T).unsqueeze(0).to(device)
                delta = x_recon - x_adv_recon
            else:
                delta = V.data + noise.data.float()
                if uniPixel:
                    delta = delta.repeat(1,3,1,1)
            x_adv_temp = x.data + delta # 3 layer update
            _ = x_adv_temp.requires_grad_()

            # calculate loss
            adv_expl, adv_acc, class_idx = get_expl(model, x_adv_temp, args.method, desired_idx=org_idx)
            loss_expl = F.mse_loss(adv_expl, target_expl)
            loss_output = F.mse_loss(adv_acc, org_acc.detach())
            loss_input = F.mse_loss(x_adv_temp, x.detach())

            loss_expl_list.append(loss_expl.item())
            loss_output_list.append(loss_output.item())
            loss_input_list.append(loss_input.item())

            # loss_diff_l2 = F.mse_loss(x_adv_temp, x.detach())
            # loss_diff_l1 = F.l1_loss(x_adv_temp, x.detach())
            total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output # + args.prefactors[2] * loss_diff_l2#  + args.prefactors[3] * loss_diff_l1
            total_loss_list[j] = total_loss.detach()
            _ = x_adv_temp.detach()
            torch.cuda.empty_cache()

            if j == 0:
                loss_expl_0 = loss_expl.item()
                loss_output_0 = loss_output.item()
                new_x_adv = x_adv_temp.clone()
                if total_loss_list[0] < best_loss:
                    best_X_adv = new_x_adv.clone().detach() # 3 layer update
                    best_loss = deepcopy(total_loss_list[0].item())

        # TODO: Change this one
        total_loss_list *= -1 # gradient ascent

        normalized_rewards = (total_loss_list - total_loss_list.mean()) / torch.clip(input=torch.std(total_loss_list), min=1e-5, max=None)
        normalized_rewards = normalized_rewards.view(-1,1).detach()
        noise_tensor = torch.stack(noise_list).view(len(noise_list),-1).detach()

        grad_log_pi = (noise_tensor.float() - mean)/std
        grad_J = torch.matmul(normalized_rewards.T, grad_log_pi).view(x_noise.shape)
        if args.MC_FGSM:
            grad_J /= std * (grad_log_pi * grad_log_pi).sum(axis=0).view(grad_J.shape)
        else:
            grad_J /= len(noise_list)
        grad_J = grad_J.detach()
        subset_idx = torch.rand(grad_J.shape) < subset_idx_threshold
        grad_J[subset_idx] = 0
        optimizer.zero_grad()

        V.grad = grad_J * (-1) # 3 layer update
        optimizer.step()
        print(scheduler.get_last_lr())
        scheduler.step()
        if args.to_compress:
            x_adv_recon = pca_3.inverse_transform_noise(V.detach().cpu().numpy().T, uniPixel)
            x_adv_recon = torch.tensor(x_adv_recon.T).unsqueeze(0).to(device)
            delta = x_recon - x_adv_recon
        else:
            delta = V
            if uniPixel:
                delta = delta.repeat(1,3,1,1)

        delta[delta < -max_delta] = -max_delta
        delta[max_delta < delta] = max_delta
        x_adv.data = x.data + delta # 3 layer update

        # clamp adversarial exmaple
        x_adv.data = clamp(x_adv.data, data_mean, data_std)

        # updating std

        if args.std_grad_update:
            grad_std = get_std_grad(normalized_rewards, noise_tensor, std.cpu().numpy(), mean.cpu().numpy(), is_scalar)
            std=std.cpu()
            std += np.clip(grad_std, a_min=-0.01, a_max=0.01)
            std=std.to(device).float()
        else:
            std *= args.std_exp_update

        std = torch.clip(std, min=0.0001)

        if i % 25 == 0:
            # if n_pop < args.max_pop:
            #     noise_list.append(noise_list[-1].clone().detach().requires_grad_())
            #     total_loss_list = torch.cat([total_loss_list, torch.tensor([0]).to(device)])
            #     n_pop += 1
            adv_expl, _, adv_idx = get_expl(model, x_adv, args.method)
            input_loss_i = F.mse_loss(x_adv, x.detach()) * args.prefactors[0]
            expl_loss_i = F.mse_loss(adv_expl, target_expl) * args.prefactors[1]
            adv_label_name = label_to_name(adv_idx.item(), args.dataset)
            path = os.path.join(args.output_dir,experiment, org_label_name, target_label_name)
            output_dir = make_dir(path)
            plot_overview([x_target, x, x_adv], [target_label_name, org_label_name, adv_label_name], [input_loss_i, expl_loss_i],
            [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}{i}_{args.method}.png")

        if args.latin_sampling:
            sample = torch.tensor(norm(loc=mean.item(), scale=std.item()).ppf(sampler.random(n=n_pop-1))).to(device)
            for k in range(1, len(noise_list)): # don't change the zero tensor
                noise_list[k].data = sample[k-1].reshape(x_noise.shape)
        else:
            for noise in noise_list[1:]: # don't change the zero tensor
                _ = noise.data.normal_(mean,std).requires_grad_()
        print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss_list[0].item(), loss_expl_0, loss_output_0))

    with open(f'loss_file/{experiment}_{index}.txt', 'a') as file:
        file.write('input loss ' + str(index) + ', ' + str(loss_input_list) + '\n')
        file.write('output loss ' + str(index) + ', ' + str(loss_output_list) + '\n')
        file.write('expl loss ' + str(index) + ', ' + str(loss_expl_list) + '\n')

    # test with original model (with relu activations)
    adv_expl, adv_acc, class_idx = get_expl(model, best_X_adv, args.method)
    adv_label_name = label_to_name(class_idx.item(), args.dataset)
    input_loss = F.mse_loss(best_X_adv, x.detach()) * args.prefactors[0]
    expl_loss = F.mse_loss(adv_expl, target_expl) * args.prefactors[1]
    # save results
    plot_overview([x_target, x, x_adv], [target_label_name, org_label_name, adv_label_name], [input_loss, expl_loss], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}best_adv_{args.method}.png")
    torch.save(x_adv, f"{output_dir}x_{args.method}.pth")


# TODO:
# PSO + Grad
# white on general porpuse net\autoencoder and than black
# choose hyper parameter search boundrys
# distance from original image