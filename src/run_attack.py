import torch.nn.functional as F
import numpy as np
import torch
import pygad
from time import time
from utils import label_to_name, get_model, get_mean_std, parse, load_random_images
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir

global org_img, org_expl, target_img, target_expl, mean_stacked, adv_acc, org_acc, args,\
    model, method, org_idx, shape, device, data_mean, data_std

def eps_selection(eps_init, gen, n_gens):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    gen = int(gen / n_gens * 100)

    if 1 < gen <= 10:
        p = eps_init / 2
    elif 10 < gen <= 20:
        p = eps_init / 4
    elif 20 < gen <= 30:
        p = eps_init / 6
    elif 30 < gen <= 40:
        p = eps_init / 8
    elif 40 < gen <= 50:
        p = eps_init / 10
    elif 50 < gen <= 60:
        p = eps_init / 12
    elif 60 < gen <= 70:
        p = eps_init / 15
    elif 70 < gen <= 80:
        p = eps_init / 20
    elif 80 < gen <= 90:
        p = eps_init / 25
    elif 90 < gen <= 100:
        p = eps_init / 30
    else:
        p = eps_init
    print(f'Alpha value: {p}, gen: {gen}')
    return p

def sparsity_selection(probs, gen, n_gens):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    gen = int(gen / n_gens * 100)

    if 10 < gen <= 20:
        p = 1 - (probs[1] / 2), probs[1] / 2
    elif 20 < gen <= 40:
        p = 1 - (probs[1] / 4), probs[1] / 4
    elif 40 < gen <= 50:
        p = 1 - (probs[1] / 6), probs[1] / 6
    elif 50 < gen <= 60:
        p = 1 - (probs[1] / 8), probs[1] / 8
    elif 60 < gen <= 70:
        p = 1 - (probs[1] / 10), probs[1] / 10
    elif 70 < gen <= 75:
        p = 1 - (probs[1] / 12), probs[1] / 12
    elif 75 < gen <= 80:
        p = 1 - (probs[1] / 15), probs[1] / 15
    elif 80 < gen <= 85:
        p = 1 - (probs[1] / 20), probs[1] / 20
    elif 85 < gen <= 90:
        p = 1 - (probs[1] / 25), probs[1] / 25
    elif 90 < gen <= 100:
        p = 1 - (probs[1] / 30), probs[1] / 30
    else:
        p = probs
    print(f'Alpha value: {p}, gen: {gen}')
    return p

def inf_norm(org_img, sol_img):
    print(np.linalg.norm((org_img.cpu().numpy().flatten() - sol_img.flatten()), ord=np.inf))

def mutation_func(offspring, ga_instance):
    offspring_shape = offspring.shape[1]
    alpha = eps_selection(1, ga_instance.generations_completed, args.num_iter)
    p = sparsity_selection([0.1, 0.9], ga_instance.generations_completed, args.num_iter)
    if ga_instance.generations_completed % 10 == 0:
        save_image(ga_instance)
    for chromosome_idx in range(offspring.shape[0]):
        rand_indices = np.random.choice([0, 1], offspring_shape, p=p)
        perturbation = np.random.choice([-alpha * args.eps, alpha * args.eps, 0], size=offspring_shape) * rand_indices
        perturbation = mean_stacked * perturbation
        perturbation += np.random.normal(0, 1 * p[1])
        # perturbation = gaussian_filter(perturbation, sigma=7)
        # perturbation = np.random.normal(0, alpha * args.eps, len(rand_indices)) * rand_indices
            # perturbation = np.random.choice([-alpha * args.eps, alpha * args.eps, 0], offspring[0].shape)
            # perturbation = np.random.choice([-alpha * args.eps, alpha * args.eps, 0], len(rand_indices)) * rand_indices
            # perturbation = np.random.uniform(-alpha * args.eps, alpha * args.eps, offspring[0].shape)

        # reshaped_offspring = offspring[chromosome_idx].reshape(3, 224, 224)

        # if np.random.random() > 0.5:
        #     reshaped_offspring[:, :, :] += perturbation
        # else:
        # if np.random.random() > 0.9:
        offspring[chromosome_idx] += perturbation
        # else:
        #     reshaped_offspring[:, :, :] += perturbation
        # else:
        #     offspring[chromosome_idx] += perturbation
        # normal_perturbation = np.random.normal(loc=0, scale=1, size=offspring[0].shape)
        # offspring[chromosome_idx] += normal_perturbation
        # offspring[chromosome_idx] = project(reshaped_offspring).reshape(offspring_shape)
        offspring[chromosome_idx] = clamp(offspring[chromosome_idx].reshape(1, 3, 224, 224), data_mean, data_std).reshape(offspring_shape)

    print(f'Gen #{ga_instance.generations_completed} mean fitness: {np.mean(ga_instance.last_generation_fitness)}')
    return offspring

def project(offspring):
    x = org_img.cpu().numpy()
    upper = x + 0.6
    lower = x - 0.6
    return np.clip(offspring, lower, upper)

def save_image(ga_instance):
    best_solution_idx = np.argmax(ga_instance.last_generation_fitness)
    x_adv = torch.tensor(ga_instance.population[best_solution_idx], dtype=torch.float).reshape(shape).to(device)
    gen = ga_instance.generations_completed
    adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, data_mean, data_std)
    plot_overview([target_img, org_img, x_adv], [target_expl, org_expl, adv_expl], data_mean, data_std,
                  filename=f"../output/overview_{args.method}_{args.pop}_{args.k}_{args.dataset}_{args.eps}_{gen}.png")

def fitness_fun(solution, solution_idx):
    global target_expl
    tensor_solution = torch.tensor(solution, dtype=torch.float).reshape(shape).to(device)
    adv_expl, adv_acc, class_idx = get_expl(model, tensor_solution, method, data_mean, data_std, desired_index=org_idx)
    loss_expl = F.mse_loss(adv_expl, target_expl)
    if args.targeted != -1:
        targeted_tensor = torch.zeros_like(org_acc.detach())
        targeted_tensor[0, args.targeted] = 1
        loss_output = F.mse_loss(adv_acc, targeted_tensor)
    else:
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
    loss_input = F.mse_loss(tensor_solution, org_img)
    fitness = 1 / (1 + (args.prefactors[0] * loss_expl + args.prefactors[1] * loss_output + args.prefactors[2] * loss_input))
    if solution_idx == 10:
        print(args.prefactors[0] * loss_expl)
    return fitness.item()

def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)

def init_pop(x, n_pop):
    pop = []
    for i in range(n_pop):
        pop.append(x.clone().cpu().detach().numpy().flatten())
    return pop

def main():
    global org_img, org_expl, target_img, target_expl, adv_acc, org_acc, args, mean_stacked
    global model, method, org_idx, shape, device, data_mean, data_std
    args = parse()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # load model
    data_mean, data_std = get_mean_std(args.dataset)
    pretrained_model = get_model(args.model, args.dataset, device)
    model = ExplainableNet(pretrained_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
    if method == ExplainingMethod.pattern_attribution:
        model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    model = model.eval().to(device)

    # load images, removed normalization
    base_images, target_images = load_random_images(args.imgs)
    for i, image_path in enumerate(base_images):
        x = load_image(args.dataset, device, image_path)
        org_img = x
        target_img = load_image(args.dataset, device, target_images[i])
        shape = x.shape

        # produce expls
        org_expl, org_acc, org_idx = get_expl(model, x, method, data_mean, data_std)
        org_expl = org_expl.detach().cpu()
        target_expl, _, target_idx = get_expl(model, target_img, method, data_mean, data_std)
        target_expl = target_expl.detach()

        # XAI mask normalized between 0-1
        mean_target_expl = target_expl.clone()
        mean_target_expl = (mean_target_expl - mean_target_expl.min()) / (mean_target_expl.max() - mean_target_expl.min())
        mean_stacked = torch.stack([mean_target_expl, mean_target_expl, mean_target_expl], dim=0).flatten().cpu().numpy()
        # mean_stacked = mean_target_expl.cpu().numpy()

        ga_instance = pygad.GA(num_generations=args.num_iter,
                               num_parents_mating=args.pop // 2,
                               parent_selection_type='tournament',
                               keep_parents=1,
                               K_tournament=args.k,
                               initial_population=init_pop(x, args.pop),
                               fitness_func=fitness_fun,
                               init_range_low=0.0,
                               crossover_type="two_points",
                               crossover_probability=0.05,
                               init_range_high=1.0,
                               stop_criteria = ["reach_0.6"] if args.dataset == 'imagenet' else ["reach_0.7"],
                               # mutation_type=mutation_func)
                               mutation_probability = [0.1, 0.01],
                               mutation_type = 'adaptive')

        start = time()
        ga_instance.run()
        end = time()
        print(end - start)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        result = torch.tensor(solution).reshape(shape)

        x_adv = torch.tensor(result, dtype=torch.float).to(device)

        # test with original model (with relu activations)
        model.change_beta(None)
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, data_mean, data_std)

        orig_label = label_to_name(org_idx.item())
        manipulated_label = label_to_name(class_idx.item())
        target_label = label_to_name(target_idx.item())
        # save results, removed std + mean from plot_overview
        output_dir = make_dir(args.output_dir)
        plot_overview([target_img, x, x_adv], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}overview_{args.method}_{args.pop}_{args.k}_{args.dataset}_{args.eps}_{orig_label}_{target_label}.png")
        print(f'Original label: {orig_label}\nmanipulated label: {manipulated_label}')
        print(f'Best fitness: {solution_fitness:.8f}')
        torch.save(x_adv, f"{output_dir}x_{args.method}.pth")
        ga_instance.plot_fitness()
        plt.savefig(f"{output_dir}fitness_{args.method}_{args.pop}_{args.k}_{args.dataset}_{args.eps}_{orig_label}_{target_label}.png")

if __name__ == "__main__":
    main()
