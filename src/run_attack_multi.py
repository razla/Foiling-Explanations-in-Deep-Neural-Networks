from string import ascii_lowercase
import torch.nn.functional as F
from random import choices
import numpy as np
import torchvision
import argparse
import torch
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2

from time import time
from utils import label_to_name, get_model, get_mean_std


from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir


global org_img, adv_expl, target_chromosome, adv_acc, org_acc, prefactors, model, method, org_idx, shape, device, data_mean, data_std, target_expl

def random_string():
    return ''.join(choices(ascii_lowercase, k=3))

def inf_norm(org_img, sol_img):
    print(np.linalg.norm((org_img.cpu().numpy().flatten() - sol_img.cpu().numpy().flatten()), ord=np.inf))



def fitness_fun(solution, solution_idx):
    global target_chromosome, adv_expl
    tensor_solution = torch.tensor(solution, dtype=torch.float).reshape(shape).to(device)
    adv_expl, adv_acc, class_idx = get_expl(model, tensor_solution, method, data_mean, data_std, desired_index=org_idx)
    target_chromosome = torch.tensor(target_chromosome).reshape(adv_expl.shape).to(device)
    loss_expl = F.mse_loss(adv_expl, target_chromosome)
    loss_output = F.mse_loss(adv_acc, org_acc.detach())
    # loss_input = F.mse_loss(tensor_solution, org_img)
    # inf_norm(tensor_solution, org_img)
    fitness = 1 / (1 + (prefactors[0] * loss_expl + prefactors[1] * loss_output)) #+ prefactors[2] * loss_input))
    if solution_idx == 5:
        print(f'Fitness: {fitness:.8f}, Expl loss: {prefactors[0] * loss_expl}, Output loss: {prefactors[1] * loss_output:.8f}') #, Input loss: {prefactors[2] * loss_input:.8f}')
    return fitness.item()

def on_fitness(ga_instance, population_fitness):
    print(population_fitness)
    print(np.mean(population_fitness))


def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)

def parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_iter', type=int, default=2000, help='number of iterations')
    argparser.add_argument('--img', type=str, default='../data/cifar10_automobile.jpg', help='image net file to run attack on')
    argparser.add_argument('--target_img', type=str, default='../data/cifar10_cat.jpg',
                           help='imagenet file used to generate target expl')
    argparser.add_argument("--model", choices=['vgg16'], default='vgg16',
                        help="specific model")
    argparser.add_argument("--dataset", choices=['imagenet', 'cifar10'], default='imagenet',
                           help="dataset")
    argparser.add_argument('--pop', type=int, default=50, help='pop size')
    argparser.add_argument('--k', type=int, default=25, help='k tournament')
    argparser.add_argument('--eps', type=float, default=0.05, help='epsilon')
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
    argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
    argparser.add_argument('--prefactors', nargs=3, default=[1e7, 1e6, 1e4], type=float,
                           help='prefactors of losses (diff expls, class loss)')
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input'],
                           default='lrp')
    args = argparser.parse_args()
    return args

def init_pop(x, n_pop):
    pop = []
    for i in range(n_pop):
        pop.append(x.clone().cpu().detach().numpy().flatten())
    return pop

def create_gene_space(x, eps):
    min_ball = torch.tile(torch.maximum(x - eps, torch.tensor(0)), (1, 1)).flatten().cpu().numpy()
    max_ball = torch.tile(torch.minimum(x + eps, torch.tensor(1)), (1, 1)).flatten().cpu().numpy()
    return min_ball, max_ball

def expl_loss(x):
    global target_expl_flattened
    tensor_solution = torch.tensor(x, dtype=torch.float).reshape(shape).to(device)
    adv_expl, adv_acc, class_idx = get_expl(model, tensor_solution, method, data_mean, data_std,
                                            desired_index=org_idx)
    loss_expl = F.mse_loss(adv_expl, target_expl).item()
    return loss_expl

def output_loss(x):
    global target_expl_flattened, org_acc
    tensor_solution = torch.tensor(x, dtype=torch.float).reshape(shape).to(device)
    adv_expl, adv_acc, class_idx = get_expl(model, tensor_solution, method, data_mean, data_std,
                                            desired_index=org_idx)
    loss_output = F.mse_loss(adv_acc, org_acc.detach()).item()
    return loss_output

def main():
    global org_img, adv_expl, target_expl, adv_acc, org_acc, prefactors, model, method, org_idx, target_chromosome, shape, device, data_mean, data_std, target_expl_flattened
    args = parse()
    prefactors = args.prefactors

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
    x = load_image(args.dataset, device, args.img)
    org_img = x
    x_target = load_image(args.dataset, device, args.target_img)
    shape = x.shape

    # produce expls
    org_expl, org_acc, org_idx = get_expl(model, x, method, data_mean, data_std)
    org_expl = org_expl.detach().cpu()
    target_expl, _, _ = get_expl(model, x_target, method, data_mean, data_std)
    target_expl = target_expl.detach()

    xl, xu = create_gene_space(x, args.eps)

    objs = [
        expl_loss,
        output_loss
    ]

    n_var = xl.flatten().shape[0]

    problem = FunctionalProblem(n_var,
                                objs,
                                xl=xl,
                                xu=xu)

    algorithm = NSGA2(pop_size=5)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=True)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()

    ga_instance = pygad.GA(num_generations=args.num_iter,
                           num_parents_mating=args.pop // 2,
                           parent_selection_type='tournament',
                           keep_parents=1,
                           K_tournament=args.k,
                           initial_population=init_pop(x, args.pop),
                           fitness_func=fitness_fun,
                           init_range_low=0.0,
                           crossover_type="two_points",
                           crossover_probability=0.01,
                           init_range_high=1.0,
                           stop_criteria = ["reach_0.5"] if args.dataset == 'imagenet' else ["reach_0.6"],
                           gene_space = gene_space,
                           on_fitness = on_fitness,
                           mutation_type="adaptive",
                           mutation_percent_genes=[0.01, 0.001],)
    start = time()
    ga_instance.run()
    end = time()
    print(end - start)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    result = gari.chromosome2img(solution, shape)

    x_adv = torch.tensor(result, dtype=torch.float).to(device)

    # test with original model (with relu activations)
    model.change_beta(None)
    adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, data_mean, data_std)

    orig_label = label_to_name(org_idx.item())
    manipulated_label = label_to_name(class_idx.item())
    # save results, removed std + mean from plot_overview
    output_dir = make_dir(args.output_dir)
    plot_overview([x_target, x, x_adv], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}overview_multi_{args.method}_{args.pop}_{args.k}.png")
    print(f'Original label: {orig_label}\nmanipulated label: {manipulated_label}')
    torch.save(x_adv, f"{output_dir}x_{args.method}.pth")


if __name__ == "__main__":
    main()
