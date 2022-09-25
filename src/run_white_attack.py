
import sys
print(sys.prefix)
from copyreg import pickle
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import torchvision
import argparse
import os.path
import torch
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR, SequentialLR
# from torchmetrics.functional import peak_signal_noise_ratio
from scipy.stats import qmc, norm
import csv

from nn.org_utils import get_expl, plot_overview, clamp, load_image, make_dir
from nn.networks import ExplainableNet
from nn.enums import ExplainingMethod

from utils import load_images, load_model, get_mean_std, label_to_name, get_optimizer
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
argparser.add_argument('--n_iter', type=int, default=1500, help='number of iterations')
argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
argparser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'], help='') #later 'cifar100' 'cifar10'
argparser.add_argument('--model', type=str, default='vgg16', help='model to use')
argparser.add_argument('--n_imgs', type=int, default=100, help='number of images to execute on')
argparser.add_argument('--img', type=str, default='../data/collie.jpeg', help='image net file to run attack on')
argparser.add_argument('--target_img', type=str, default='../data/tiger_cat.jpeg',
                        help='imagenet file used to generate target expl')
argparser.add_argument('--output_dir', type=str, default='output/', help='directory to save results to')
argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
argparser.add_argument('--optimizer', help='', choices=['Adam', 'SGD', 'RMSprop'], type=str, default='Adam')
argparser.add_argument('--prefactors', nargs=4, default=[1e11, 1e6, 1e4, 1e2], type=float,
                        help='prefactors of losses (diff expls, class loss, l2 loss, l1 loss)')
argparser.add_argument('--method', help='algorithm for expls',
                        choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                'pattern_attribution', 'grad_times_input'],
                        default='lrp')

args = argparser.parse_args()



experiment = f'_method_{args.method}_model_{args.model}_data_{args.dataset}'
seed = 0
experiment += f'_seed_{seed}'
np.save('argparser_white/' + experiment + '.npy', args.__dict__, allow_pickle=True)
print(experiment, flush=True)

# experiment = 'debug_white'
# print(experiment)

# options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
method = getattr(ExplainingMethod, args.method)

# load model
data_mean, data_std = get_mean_std(args.dataset)
pretrained_model = load_model(args.model, args.dataset, device)
model = ExplainableNet(pretrained_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
if method == ExplainingMethod.pattern_attribution:
    model.load_state_dict(torch.load('/cs_storage/public_datasets/model_vgg16_pattern_small.pth'), strict=False)
model = model.eval().to(device)


base_images_paths, target_images_paths = load_images(args.n_imgs, args.dataset, seed)
for index, (base_image, target_image) in enumerate(zip(base_images_paths, target_images_paths)):
    loss_expl_list = []
    loss_input_list = []
    loss_output_list = []
    # load images
    x = load_image(data_mean, data_std, device, base_image, args.dataset)
    x_target = load_image(data_mean, data_std, device, target_image, args.dataset)
    if x is None or x_target is None:
        continue
    x_adv = x.clone().detach().requires_grad_()
    # produce expls
    org_expl, org_acc, org_idx = get_expl(model, x, method)
    org_expl = org_expl.detach().cpu()
    org_label_name = label_to_name(org_idx.item())
    target_expl, _, target_idx = get_expl(model, x_target, method)
    target_expl = target_expl.detach()
    target_label_name = label_to_name(target_idx.item())

    best_X_adv = x_adv.clone()
    best_loss = float('inf')

    optimizer = torch.optim.Adam([x_adv], lr=args.lr)

    for i in range(args.n_iter):
        if args.beta_growth:
            model.change_beta(get_beta(i, args.n_iter))

        optimizer.zero_grad()
        # calculate loss
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=org_idx)
        loss_expl = F.mse_loss(adv_expl, target_expl)
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
        loss_input = F.mse_loss(x_adv, x.detach())
        total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output
        
        # update adversarial example
        total_loss.backward()
        optimizer.step()
        x_adv.data = clamp(x_adv.data, data_mean, data_std)

        loss_expl_list.append(loss_expl.item())
        loss_output_list.append(loss_output.item())
        loss_input_list.append(loss_input.item())

        loss_expl_0 = loss_expl.item()
        loss_output_0 = loss_output.item()
        if total_loss < best_loss:
            best_X_adv = x_adv.clone().detach() # 3 layer update
            best_loss = deepcopy(total_loss.item())


        if i % 25 == 0:
            adv_expl, _, adv_idx = get_expl(model, x_adv, method)
            input_loss_i = F.mse_loss(x_adv, x.detach()) * args.prefactors[0]
            expl_loss_i = F.mse_loss(adv_expl, target_expl) * args.prefactors[1]
            adv_label_name = label_to_name(adv_idx.item())
            path = os.path.join(args.output_dir,experiment, org_label_name, target_label_name)
            output_dir = make_dir(path)
            plot_overview([x_target, x, x_adv], [target_label_name, org_label_name, adv_label_name], [input_loss_i, expl_loss_i], 
            [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}{i}_{args.method}.png")

        print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss.item(), loss_expl_0, loss_output_0))

    with open(f'loss_file_white/{experiment}_{index}.txt', 'a') as file:
        file.write('input loss ' + str(index) + ', ' + str(loss_input_list) + '\n')
        file.write('output loss ' + str(index) + ', ' + str(loss_output_list) + '\n')
        file.write('expl loss ' + str(index) + ', ' + str(loss_expl_list) + '\n')

    # test with original model (with relu activations)
    model.change_beta(None)
    adv_expl, adv_acc, class_idx = get_expl(model, best_X_adv, method)
    adv_label_name = label_to_name(class_idx.item())
    input_loss = F.mse_loss(best_X_adv, x.detach()) * args.prefactors[0]
    expl_loss = F.mse_loss(adv_expl, target_expl) * args.prefactors[1]
    # save results
    plot_overview([x_target, x, best_X_adv], [target_label_name, org_label_name, adv_label_name], [input_loss, expl_loss], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}best_adv_{args.method}.png")
    torch.save(best_X_adv, f"{output_dir}x_{args.method}.pth")


# if __name__ == "__main__":
#     main()
