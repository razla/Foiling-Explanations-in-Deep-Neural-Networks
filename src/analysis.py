
from cProfile import label
from copyreg import pickle
from stringprep import c22_specials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from operator import add
# argparser_names = listdir('argparser')
argparser_names = listdir('/cs_storage/public_datasets/results/argparser')
file = argparser_names[-1]
res = []
headers = ['n_iter', 'n_pop', 'method', 'lr', 'max_delta', 'optimizer', 'weight_decay',
 'compression_type', 'n_components', 'LS', 'SYN','MC_FGSM', 'uniPixel', 'momentum',
#    'pair_index',
    'input_loss', 'output_loss', 'expl_loss',
    'input_loss_min', 'output_loss_min', 'expl_loss_min']
for file in argparser_names:
    # argparser_dict = np.load(join('argparser', file), allow_pickle=True).item()
    argparser_dict = np.load(join('/cs_storage/public_datasets/results/argparser', file), allow_pickle=True).item()
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
    tmp+=[[],[],[],[],[],[]]
    for i in range(100):
        try: # not all images ready
            # with open(f'loss_file/{experiment}_{i}.txt') as file:
            with open(f'/cs_storage/public_datasets/results/loss_file/{experiment}_{i}.txt') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                input_loss = eval('['+lines[0].split('[')[1])[1:]
                output_loss = eval('['+lines[1].split('[')[1])[1:]
                expl_loss = eval('['+lines[2].split('[')[1])[1:]
                # res.append(tmp + [i, min(input_loss), min(output_loss), min(expl_loss)])
                # res.append(tmp + [i, min(input_loss), min(output_loss), min(expl_loss)])
                if tmp[-6] == []:
                    tmp[-6]=np.array(input_loss)
                    tmp[-5]=np.array(output_loss)
                    tmp[-4]=np.array(expl_loss)
                else:
                    tmp[-6]+=input_loss
                    tmp[-5]+=output_loss
                    tmp[-4]+=expl_loss
                tmp[-3].append(min(input_loss))
                tmp[-2].append(min(output_loss))
                tmp[-1].append(min(expl_loss))
        except:
            pass
    res.append(tmp)
frame = pd.DataFrame(res,columns=headers)

frame[frame['LS'] == True]['output_loss']


frame.pivot(index = 'method', columns='output_loss')

frame.pivot_table(index ='lr', columns ='method', values = 'input_loss')




# 'input_loss', 'output_loss', 'expl_loss'
import matplotlib.pyplot as plt
for i, method in enumerate(frame['method'].unique()):
    frame_method = frame[frame['method'] == method]
    tmp = []
    for loss_list in frame_method['expl_loss_min']:
       tmp+=loss_list
    plt.bar(i, height=np.mean(tmp), label=method)
plt.legend()
plt.savefig('test.png')
plt.close()



## 

attack_order = choices=['lrp', 'guided_backprop', 'gradient',# 'integrated_grad',
'pattern_attribution', 'grad_times_input']


from collections import defaultdict
frame_dict = defaultdict(list)
frame_ = frame[['n_pop', 'method', 'lr', 'LS', 'input_loss_min', 'output_loss_min', 'expl_loss_min']]

for index, f in frame_.iterrows():
    title = f"n_pop={f['n_pop']}, lr={f['lr']}, LS={f['LS']}"
    frame_dict[(f['n_pop'], f['lr'], f['LS'])].append([(title, f['method'], f['expl_loss_min'])])

len(frame_dict.keys())

fig, axs = plt.subplots(3, 3,figsize=(20, 20), sharey='all')
for i, key in enumerate(frame_dict.keys()):
    boxplot_data = []
    boxplot_labels = []
    boxplot_label_order = []
    for _, f in enumerate(frame_dict[key]):
        boxplot_data.append(np.array(f[0][2]))
        boxplot_labels.append(f[0][1])
        boxplot_label_order.append(attack_order.index(f[0][1]))
    bp_dict = fig.axes[i].boxplot(boxplot_data, labels=boxplot_labels, positions=boxplot_label_order, meanline=True, showmeans=True)
    fig.axes[i].set_title(f[0][0])
    plt.setp(fig.axes[i].get_xticklabels(), rotation=15)

    for k, line in enumerate(bp_dict['means']):
        x, y = line.get_xydata()[1] # top of median line
        fig.axes[i].text(x, y, "{:.2e}".format(np.mean(boxplot_data[k])), horizontalalignment='right')#, verticalalignment='bottom') # draw above, centered


fig.suptitle('expl_loss_min',size=50)
plt.savefig('test5.png')
plt.show()
plt.close()





##


from collections import defaultdict
frame_dict = defaultdict(list)
frame_ = frame[['n_pop', 'method', 'lr', 'LS', 'input_loss', 'output_loss', 'expl_loss']]

for index, f in frame_.iterrows():
    title = f"n_pop={f['n_pop']}, lr={f['lr']}, LS={f['LS']}"
    frame_dict[(f['n_pop'], f['lr'], f['LS'])].append([(title, f['method'], f['expl_loss'])])

len(frame_dict.keys())

fig, axs = plt.subplots(3, 3,figsize=(18, 18), sharey='all')
for i, key in enumerate(frame_dict.keys()):
    boxplot_data = []
    boxplot_labels = []
    boxplot_label_order = []
    for _, f in enumerate(frame_dict[key]):
        # boxplot_data.append(np.array(f[0][2])[::key[0]])
        data = np.array(f[0][2])[(key[0]-1)::key[0]]
        bp_dict = fig.axes[i].plot(np.arange(data.shape[0]), data, label=f[0][1], color='C'+str(attack_order.index(f[0][1])))
    fig.axes[i].set_title(f[0][0])
    fig.axes[i].legend()
    plt.setp(fig.axes[i].get_xticklabels(), rotation=15)



import matplotlib.lines as mlines
# red_patch = mpatches.Patch(color='red', label='The red data')
# blue_patch = mpatches.Patch(color='blue', label='The blue data')

lines = [mlines.Line2D([], [], color='C{a}'.format(), markersize=15, label='Blue stars') for a in range(len(attack_order))]
fig.legend(handles=lines)

# fig.legend(fig.axes[i], labels=attack_order)
fig.suptitle('expl_loss',size=50)
plt.savefig('test_4.png')
plt.show()
plt.close()











# src = /home/snirvit/AttaXAI/src/argparser
# dest = /cs_storage/public_datasets/results
# cp -r /home/snirvit/AttaXAI/src/argparser /cs_storage/public_datasets/results
# cp -r /home/snirvit/AttaXAI/src/loss_file /cs_storage/public_datasets/results