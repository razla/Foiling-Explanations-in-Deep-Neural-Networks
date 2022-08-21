import argparse
from copy import deepcopy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import torchvision
import torch.nn.functional as F
import numpy as np

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.org_utils import get_expl, plot_overview, clamp, load_image, make_dir


def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)


# def main():
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=1500, help='number of iterations')
argparser.add_argument('--img', type=str, default='../data/collie.jpeg', help='image net file to run attack on')
argparser.add_argument('--target_img', type=str, default='../data/tiger_cat.jpeg',
                        help='imagenet file used to generate target expl')
argparser.add_argument('--lr', type=float, default=0.0002, help='lr')
argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
argparser.add_argument('--prefactors', nargs=2, default=[1e11, 1e6], type=float,
                        help='prefactors of losses (diff expls, class loss)')
argparser.add_argument('--method', help='algorithm for expls',
                        choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                'pattern_attribution', 'grad_times_input'],
                        default='lrp')
args = argparser.parse_args()

# options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
method = getattr(ExplainingMethod, args.method)

# load model
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
vgg_model = torchvision.models.vgg16(pretrained=True)
model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
if method == ExplainingMethod.pattern_attribution:
    model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
model = model.eval().to(device)

# load images
x = load_image(data_mean, data_std, device, args.img)
x_target = load_image(data_mean, data_std, device, args.target_img)
x_adv = x.clone().detach().requires_grad_()
x_adv_temp = x.clone().detach().requires_grad_()

# produce expls
org_expl, org_acc, org_idx = get_expl(model, x, method)
org_expl = org_expl.detach().cpu()
target_expl, _, _ = get_expl(model, x_target, method)
target_expl = target_expl.detach()


pop_size = 64
std = 0.1
lr = 0.2
mu = 0.0
l2_W = 250.0
l1_W = 250.0

total_loss_list = torch.Tensor(pop_size).to(device)
noise_list = [x.clone().detach().data.normal_(0,std).requires_grad_() for _ in range(pop_size)]
noise_list[0] = x.clone().detach().zero_().requires_grad_()
V = x.clone().detach().zero_()
best_X_adv = deepcopy(x_adv)
best_loss = float('inf')

for i in range(args.num_iter):
    if args.beta_growth:
        model.change_beta(get_beta(i, args.num_iter))
    
    loss_expl_0 = None
    loss_output_0 = None
    for j, noise in enumerate(noise_list):
        x_adv_temp = x_adv.data + noise.data
        _ = x_adv_temp.requires_grad_()
        # calculate loss
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv_temp, method, desired_index=org_idx)
        loss_expl = F.mse_loss(adv_expl, target_expl)
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
        loss_diff_l2 = F.mse_loss(x_adv_temp, x.detach())
        loss_diff_l1 = F.l1_loss(x_adv_temp, x.detach())
        total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output + l2_W * loss_diff_l2 + l1_W * loss_diff_l1
        total_loss_list[j] = total_loss.detach()
        _ = x_adv_temp.detach()
        torch.cuda.empty_cache()
        if j == 0:
            loss_expl_0 = loss_expl.item()
            loss_output_0 = loss_output.item()

    if total_loss_list[0] < best_loss: 
        best_X_adv = deepcopy(x_adv.data).detach()
        best_loss = deepcopy(total_loss_list[0].item())

    total_loss_list *= -1 # gradient ascent
    normalized_rewards = (total_loss_list - total_loss_list.mean()) / torch.clip(input=torch.std(total_loss_list), min=1e-5, max=None)
    normalized_rewards = normalized_rewards.view(-1,1).detach()
    noise_tensor = torch.stack(noise_list).view(len(noise_list),-1).detach()

    grad_log_pi = (noise_tensor - 0)/std
    grad_J = torch.matmul(normalized_rewards.T, grad_log_pi).view(x_adv.data.shape)
    grad_J /= len(noise_list)
    grad_J = grad_J.detach()
    lr *= 0.999
    mu *= 0.999
    std *= 0.9995
    V = mu*V + lr * grad_J 
    x_adv.data = x_adv.data + V
    
    for noise in noise_list[1:]: # don't change the zero tensor
         _ = noise.data.normal_(0,std).requires_grad_()

    # clamp adversarial example
    # Note: x_adv.data returns tensor which shares data with x_adv but requires
    #       no gradient. Since we do not want to differentiate the clamping,
    #       this is what we need
    x_adv.data = clamp(x_adv.data, data_mean, data_std)
    print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss_list[0].item(), loss_expl_0, loss_output_0))

# test with original model (with relu activations)
model.change_beta(None)
# adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method)
adv_expl, adv_acc, class_idx = get_expl(model, best_X_adv, method)


# save results
args.output_dir = args.output_dir[3:]
output_dir = make_dir(args.output_dir)
plot_overview([x_target, x, x_adv], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}l1_250_l2_250_0_momentum_{args.method}.png")
torch.save(x_adv, f"{output_dir}x_{args.method}.pth")


# if __name__ == "__main__":
#     main()
