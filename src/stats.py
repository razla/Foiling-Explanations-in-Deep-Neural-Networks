

import numpy as np
import torch


def get_std_grad(normalized_rewards, noise_tensor, std, mean, is_scalar=True):
    grad_std = 0 
    for k in range(1,noise_tensor.shape[0]):
        Xk = noise_tensor[k].detach().cpu().numpy().reshape(1,-1)
        grad_K = normalized_rewards[k].item() * (-1) * ((std**2) - np.square(Xk) + 2*np.dot(mean, Xk.T).sum()  - np.square(mean))/(std**3)
        if is_scalar:
            grad_K = grad_K.mean()
        grad_std += grad_K
    grad_std /= (noise_tensor.shape[0]-1)
    return grad_std


def get_std_grad_scalar(normalized_rewards, noise_tensor, std, mean):
    grad_std = 0
    for k in range(1,noise_tensor.shape[0]):
        Xk = noise_tensor[k].detach().cpu().numpy()
        grad_std += np.mean(normalized_rewards[k].item()* (-1) * ((std**2) - np.square(Xk) + 2*np.dot(Xk, mean) - np.square(mean))/(std**3))
    grad_std /= (noise_tensor.shape[0]-1)
    return grad_std



def get_mean_grad_scalar(normalized_rewards, noise_tensor, std, mean):
    mean_grad = 0
    for k in range(1,noise_tensor.shape[0]):
        Xk = noise_tensor[k].detach().cpu().numpy()
        mean_grad += np.mean(normalized_rewards[k].item() * (Xk - mean)/(std**2))
    mean_grad /= (noise_tensor.shape[0]-1)
    return mean_grad



