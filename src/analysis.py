
from copyreg import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

argparser_names = listdir('argparser')
file = argparser_names[-1]
res = []
headers = ['n_iter', 'n_pop', 'method', 'lr', 'max_delta', 'optimizer', 'weight_decay',
 'compression_type', 'n_components', 'LS', 'SYN','MC_FGSM', 'uniPixel',
  'momentum', 'pair_index', 'input_loss', 'output_loss', 'expl_loss']
for file in argparser_names:
    argparser_dict = np.load(join('argparser', file), allow_pickle=True).item()
    locals().update(argparser_dict) # turn dict key,val to local variables
    experiment = f'n_iter_{n_iter}_n_pop_{n_pop}_method_{method}_lr_{lr}_max_delta_{max_delta}_opt_{optimizer}_w_decay_{weight_decay}'
    tmp = [n_iter, n_pop, method, lr, max_delta, optimizer, weight_decay]
    if to_compress:
        if compression_method.lower() == 'pca':
            experiment+='_PCA'
            tmp.append('PCA')
        else:
            experiment+='_SVD'
            tmp.append('SVD')
        experiment += f'_{n_components}'
        tmp.append(n_components)
    else:
        tmp+=[None, None]
    if latin_sampling:
        experiment += f'_LS'
        tmp.append(True)
    else:
        tmp.append(False)
    if synthesize:
        experiment += f'_SYN'
        tmp.append(True)
    else:
        tmp.append(False)
    if MC_FGSM:
        experiment += f'_MC_FGSM'
        tmp.append(True)
    else:
        tmp.append(False)
    if uniPixel:
        experiment += f'_uniPixel'
        tmp.append(True)
    else:
        tmp.append(False)
    if optimizer.lower() in ['sgd', 'rmsprop']:
        experiment += f'_mu_{momentum}'
        tmp.append(momentum)
    else:
        tmp.append(None)
    if is_scalar:
        experiment += f'_scalar_std'
    if std_grad_update:
        experiment += f'_std_grad_update'
    else:
        experiment += f'_std_exp_update_{std_exp_update}'

    experiment += f'_prefactors_{str(prefactors)}'
    seed = 0
    experiment += f'_seed_{seed}'
    for i in range(100):
        try: # not all images ready
            with open(f'loss_file/{experiment}_{i}.txt') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                input_loss = eval('['+lines[0].split('[')[1])[1:]
                output_loss = eval('['+lines[1].split('[')[1])[1:]
                expl_loss = eval('['+lines[2].split('[')[1])[1:]
                res.append(tmp + [i, min(input_loss), min(output_loss), min(expl_loss)])
        except:
            pass

frame = pd.DataFrame(res,columns=headers)



