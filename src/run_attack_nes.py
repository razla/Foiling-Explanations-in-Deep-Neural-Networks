import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import torchvision
import argparse
import os.path
import torch

from nn.org_utils import get_expl, plot_overview, clamp, load_image, make_dir
from nn.networks import ExplainableNet
from nn.enums import ExplainingMethod

from utils import load_images, get_mean_std, label_to_name
from compression import PCA_3_channels
from stats import get_std_grad

def get_beta(i, n_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / n_iter)

# def main():
argparser = argparse.ArgumentParser()
argparser.add_argument('--n_iter', type=int, default=2, help='number of iterations')
argparser.add_argument('--n_pop', type=int, default=100, help='number of individuals sampled from gaussian')
argparser.add_argument('--max_pop', type=int, default=100, help='maximum size of population')
argparser.add_argument('--mean', type=float, default=0, help='mean of the gaussian distribution')
argparser.add_argument('--std', type=float, default=0.1, help='std of the gaussian distribution')
argparser.add_argument('--lr', type=float, default=0.1, help='learning rate')
argparser.add_argument('--momentum', type=float, default=0, help='momentum constant')
argparser.add_argument('--dataset', type=str, default='imagenet', help='dataset to execute on')
argparser.add_argument('--n_imgs', type=int, default=1, help='number of images to execute on')
argparser.add_argument('--img', type=str, default='../data/collie.jpeg', help='image net file to run attack on')
argparser.add_argument('--target_img', type=str, default='../data/tiger_cat.jpeg',
                        help='imagenet file used to generate target expl')
argparser.add_argument('--output_dir', type=str, default='output/', help='directory to save results to')
argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
argparser.add_argument('--is_scalar', help='is std a scalar', type=bool, default=True)
argparser.add_argument('--is_PCA', help='applying PCA', type=bool, default=True)
argparser.add_argument('--PCA_n_components', help='How many principle components', type=int, default=50)

argparser.add_argument('--prefactors', nargs=4, default=[1e11, 1e6, 1e3, 1e2], type=float,
                        help='prefactors of losses (diff expls, class loss, l2 loss, l1 loss)')
argparser.add_argument('--method', help='algorithm for expls',
                        choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                'pattern_attribution', 'grad_times_input'],
                        default='lrp')

args = argparser.parse_args()

n_pop = args.n_pop
mean = torch.tensor(args.mean)
std = torch.tensor(args.std)
lr = args.lr
mu = args.momentum
is_scalar = args.is_scalar

# options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
method = getattr(ExplainingMethod, args.method)

# load model
data_mean, data_std = get_mean_std(args.dataset)
vgg_model = torchvision.models.vgg16(pretrained=True)
model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
if method == ExplainingMethod.pattern_attribution:
    model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
model = model.eval().to(device)

base_images_paths, target_images_paths = load_images(args.n_imgs)
for base_image, target_image in zip(base_images_paths, target_images_paths):
    # load images
    x = load_image(data_mean, data_std, device, base_image)
    x_target = load_image(data_mean, data_std, device, target_image)
    x_adv = x.clone().detach().requires_grad_()
    x_noise = x.clone().detach().requires_grad_()
    # produce expls
    org_expl, org_acc, org_idx = get_expl(model, x, method)
    org_expl = org_expl.detach().cpu()
    org_label_name = label_to_name(org_idx.item())
    target_expl, _, target_idx = get_expl(model, x_target, method)
    target_expl = target_expl.detach()
    target_label_name = label_to_name(target_idx.item())

    total_loss_list = torch.Tensor(n_pop).to(device)
    best_X_adv = deepcopy(x_adv)
    best_loss = float('inf')
    if args.is_PCA:
        pca_3 = PCA_3_channels(args.PCA_n_components)
        x_compressed = pca_3.fit_transform(x.detach().cpu().numpy()[0].T)
        img_recon = pca_3.inverse_transform()
        x_recon = torch.tensor(img_recon.T).unsqueeze(0).to(device)
        x_compressed = torch.tensor(x_compressed.T).unsqueeze(0).to(device)
        x_adv_comp = x_compressed.clone()
        x_noise = x_compressed.clone()
    noise_list = [x_noise.clone().detach().data.normal_(mean.item(), std.item()).requires_grad_() for _ in range(n_pop)]
    noise_list[0] = x_noise.clone().detach().zero_().requires_grad_()
    V = x_noise.clone().detach().zero_()


    for i in range(args.n_iter):
        if args.beta_growth:
            model.change_beta(get_beta(i, args.n_iter))

        loss_expl_0 = None
        loss_output_0 = None
        for j, noise in enumerate(noise_list):
            if args.is_PCA:
                x_adv_recon = pca_3.inverse_transform_noise(noise.data.T.cpu().numpy())
                x_adv_recon = torch.tensor(x_adv_recon.T).unsqueeze(0).to(device)
                x_adv_temp = x_adv.data + x_recon - x_adv_recon
            else:
                x_adv_temp = x_adv.data + noise.data
            _ = x_adv_temp.requires_grad_()

            # calculate loss
            adv_expl, adv_acc, class_idx = get_expl(model, x_adv_temp, method, desired_index=org_idx)
            loss_expl = F.mse_loss(adv_expl, target_expl)
            loss_output = F.mse_loss(adv_acc, org_acc.detach())
            # loss_diff_l2 = F.mse_loss(x_adv_temp, x.detach())
            # loss_diff_l1 = F.l1_loss(x_adv_temp, x.detach())
            total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output # + args.prefactors[2] * loss_diff_l2 # + args.prefactors[3] * loss_diff_l1
            total_loss_list[j] = total_loss.detach()
            _ = x_adv_temp.detach()
            torch.cuda.empty_cache()

            if j == 0:
                loss_expl_0 = loss_expl.item()
                loss_output_0 = loss_output.item()

        if total_loss_list[0] < best_loss:
            best_X_adv = deepcopy(x_adv.data).detach()
            best_loss = deepcopy(total_loss_list[0].item())

        # TODO: Change this one
        total_loss_list *= -1 # gradient ascent
        normalized_rewards = (total_loss_list - total_loss_list.mean()) / torch.clip(input=torch.std(total_loss_list), min=1e-5, max=None)
        normalized_rewards = normalized_rewards.view(-1,1).detach()
        noise_tensor = torch.stack(noise_list).view(len(noise_list),-1).detach()

        grad_log_pi = (noise_tensor - mean)/std
        grad_J = torch.matmul(normalized_rewards.T, grad_log_pi).view(x_noise.shape)
        grad_J /= len(noise_list)
        grad_J = grad_J.detach()
        lr *= 0.9999
        mu *= 0.9999
        V = mu*V + lr * grad_J
        if args.is_PCA:
            x_adv_recon = pca_3.inverse_transform_noise(V.detach().cpu().numpy().T)
            x_adv_recon = torch.tensor(x_adv_recon.T).unsqueeze(0).to(device)
            x_adv.data = x_adv.data + x_recon - x_adv_recon
        else:
            x_adv.data = x_adv.data + V
        




        # updating std
        grad_std = get_std_grad(normalized_rewards, noise_tensor, std.cpu().numpy(), mean.cpu().numpy(), is_scalar)
        std=std.cpu()
        std += np.clip(grad_std, a_min=-0.01, a_max=0.01)
        std=std.to(device).float()
        std = torch.clip(std, min=0.0001)

        if i % 25 == 0:
            if n_pop < args.max_pop:
                noise_list.append(noise_list[-1].clone().detach().requires_grad_())
                total_loss_list = torch.cat([total_loss_list, torch.tensor([0]).to(device)])
                n_pop += 1
            adv_expl, _, adv_idx = get_expl(model, x_adv, method)
            input_loss_i = F.mse_loss(x_adv, x.detach())
            expl_loss_i = F.mse_loss(adv_expl, target_expl)
            adv_label_name = label_to_name(adv_idx.item())
            path = os.path.join(args.output_dir, org_label_name, target_label_name)
            output_dir = make_dir(path)
            plot_overview([x_target, x, x_adv], [target_label_name, org_label_name, adv_label_name], [input_loss_i, expl_loss_i], 
            [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}{i}_{args.method}.png")

        for noise in noise_list[1:]: # don't change the zero tensor
                _ = noise.data.normal_(mean,std).requires_grad_()

        # clamp adversarial exmaple
        x_adv.data = clamp(x_adv.data, data_mean, data_std)

        print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss_list[0].item(), loss_expl_0, loss_output_0))

# test with original model (with relu activations)
model.change_beta(None)
adv_expl, adv_acc, class_idx = get_expl(model, best_X_adv, method)
adv_label_name = label_to_name(class_idx.item())
input_loss = F.mse_loss(best_X_adv, x.detach())
expl_loss = F.mse_loss(adv_expl, target_expl)
# save results
plot_overview([x_target, x, x_adv], [target_label_name, org_label_name, adv_label_name], [input_loss, expl_loss], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}best_adv_{args.method}.png")
torch.save(x_adv, f"{output_dir}x_{args.method}.pth")

# if __name__ == "__main__":
#     main()