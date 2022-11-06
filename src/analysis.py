
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
# argparser_names = listdir('/cs_storage/public_datasets/results/argparser')
# argparser_names = listdir('argparser_3')
# argparser_names = listdir('/cs_storage/public_datasets/results/argparser_3')
# argparser_names = listdir('/cs_storage/public_datasets/results/cifar_VGG/argparser_VGG_CIFAR')
# argparser_names = listdir('/cs_storage/public_datasets/results/cifar_VGG/loss_file_VGG_CIFAR')


# argparser_path = '/cs_storage/public_datasets/results/cifar_VGG/argparser_VGG_CIFAR'
# argparser_path = '/cs_storage/public_datasets/results/cifar_vgg_new2/argparser_VGG_CIFAR'
# argparser_path = '/cs_storage/public_datasets/results/cifar_REPVGG_results_2/argparser_REPVGG_CIFAR'
argparser_path = '/cs_storage/public_datasets/results/imagenet_inception1/argparser_INET_INC'
# loss_path = '/cs_storage/public_datasets/results/cifar_VGG/loss_file_VGG_CIFAR'
# loss_path = '/cs_storage/public_datasets/results/cifar_vgg_new2/loss_file_VGG_CIFAR'
# loss_path = '/cs_storage/public_datasets/results/cifar_REPVGG_results_2/loss_file_REPVGG_CIFAR'
loss_path = '/cs_storage/public_datasets/results/imagenet_inception1/loss_file_INET_INC'
argparser_names = listdir(argparser_path)
file = argparser_names[-1]
res = []
# headers = ['n_iter', 'n_pop', 'method', 'lr', 'max_delta', 'optimizer', 'weight_decay',
#  'compression_type', 'n_components', 'LS', 'SYN','MC_FGSM', 'uniPixel', 'momentum',
# #    'pair_index',
#     'input_loss', 'output_loss', 'expl_loss',
#     'input_loss_min', 'output_loss_min', 'expl_loss_min']

headers = ['model', 'dataset', 'n_iter', 'n_pop', 'method', 'lr', 'max_delta', 'optimizer', 'weight_decay',
           'lr_decay','prefactors', 'LS', 'input_loss', 'output_loss', 'expl_loss',
           'input_loss_min', 'output_loss_min', 'expl_loss_min']



for file in argparser_names:
    argparser_dict = np.load(join(argparser_path, file), allow_pickle=True).item()
    # argparser_dict = np.load(join('/cs_storage/public_datasets/results/argparser_3', file), allow_pickle=True).item()
    # argparser_dict = np.load(join('argparser_3', file), allow_pickle=True).item()
    # argparser_dict = np.load(join('/cs_storage/public_datasets/results/argparser', file), allow_pickle=True).item()
    locals().update(argparser_dict) # tsurn dict key,val to local variables
    # experiment = f'n_iter_{n_iter}_n_pop_{n_pop}_method_{method}_lr_{lr}_max_delta_{max_delta}_opt_{optimizer}_w_decay_{weight_decay}'
    experiment = f'model_{model}_dataset_{dataset}_n_iter_{n_iter}_n_pop_{n_pop}_method_{method}_lr_{lr}_max_delta_{max_delta}_opt_{optimizer}_w_decay_{weight_decay}_lr_decay_{lr_decay}'

    tmp = [model, dataset, n_iter, n_pop, method, lr, max_delta, optimizer, weight_decay, lr_decay, prefactors]
    # if to_compress:
    #     if compression_method.lower() == 'pca':
    #         experiment+='_PCA'
    #         tmp.append('PCA')
    #     else:
    #         experiment+='_SVD'
    #         tmp.append('SVD')
    #     experiment += f'_{n_components}'
    #     tmp.append(n_components)
    # else:
        # tmp+=[None, None]
    if latin_sampling:
        experiment += f'_LS'
        tmp.append(True)
    else:
        tmp.append(False)
    # if synthesize:
    #     experiment += f'_SYN'
    #     tmp.append(True)
    # else:
        # tmp.append(False)
    # if MC_FGSM:
    #     experiment += f'_MC_FGSM'
    #     tmp.append(True)
    # else:
        # tmp.append(False)
    # if uniPixel:
    #     experiment += f'_uniPixel'
    #     tmp.append(True)
    # else:
        # tmp.append(False)
    # if optimizer.lower() in ['sgd', 'rmsprop']:
    #     experiment += f'_mu_{momentum}'
    #     tmp.append(momentum)
    # else:
        # tmp.append(None)
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
    for i in range(103):
        try: # not all images ready
            with open(loss_path + f'/{experiment}_{i}.txt') as file:
            # with open(f'/cs_storage/public_datasets/results/loss_file_3/{experiment}_{i}.txt') as file:
            # with open(f'loss_file_3/{experiment}_{i}.txt') as file:
            # with open(f'/cs_storage/public_datasets/results/loss_file/{experiment}_{i}.txt') as file:
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
                idx_min = np.argmin(expl_loss)
                tmp[-3].append(input_loss[idx_min])
                tmp[-2].append(output_loss[idx_min])
                tmp[-1].append(expl_loss[idx_min])
        except:
            pass
            # print(experiment, i)
            # print(experiment, i)
    if len(tmp[-3]) > 0:
        tmp[-6] /= len(tmp[-3])
        tmp[-5] /= len(tmp[-3])
        tmp[-4] /= len(tmp[-3])
    res.append(tmp)
frame = pd.DataFrame(res,columns=headers)


frame['expl_loss_min'].apply(len)

idx = np.where(frame['expl_loss_min'].apply(len) < 100)[0]

frame.iloc[idx,[2,3,4,5,11]]

frame[frame['LS'] == True]['output_loss']

frame['expl_loss_min'].apply(len)


idx = np.where(frame['prefactors'].apply(sum) >= 100000000000.0)[0]

frame.iloc[idx,[2,3,4,5,11]]



# frame.pivot(index = 'method', columns='output_loss')

# frame.pivot_table(index ='lr', columns ='method', values = 'input_loss')




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

##
import matplotlib.pyplot as plt

font_size = 25
labelsize = 30
axes_size = 30
BIGGER_SIZE = 50

plt.rc('font', size=font_size)          # controls default text sizes
plt.rc('axes', titlesize=axes_size)     # fontsize of the axes title
plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('axes.titlesize', titlesize=BIGGER_SIZE)  # fontsize of the figure title



## expl_loss_min

attack_order = choices=['lrp', 'guided_backprop', 'gradient',# 'integrated_grad',
# 'pattern_attribution', 
'grad_times_input', 'deep_lift']


from collections import defaultdict
frame_dict = defaultdict(list)
frame_ = frame[['n_pop', 'method', 'lr', 'LS', 'input_loss_min', 'output_loss_min', 'expl_loss_min']]

for index, f in frame_.iterrows():
    title = f"n_pop={f['n_pop']}, lr={f['lr']}, LS={f['LS']}"
    frame_dict[(f['n_pop'], f['lr'], f['LS'])].append([(title, f['method'], f['expl_loss_min'])])

len(frame_dict.keys())

fig, axs = plt.subplots(4, 2,figsize=(27, 27), sharey='all')
# axs[-1,-1].set_axis_off()

for i, key in enumerate(frame_dict.keys()):
    boxplot_data = []
    boxplot_labels = []
    boxplot_label_order = []
    for _, f in enumerate(frame_dict[key]):
        boxplot_data.append(np.array(f[0][2]))
        boxplot_labels.append(f[0][1])
        boxplot_label_order.append(attack_order.index(f[0][1]))
    bp_dict = fig.axes[i].boxplot(boxplot_data, labels=boxplot_labels, positions=boxplot_label_order, meanline=True, showmeans=True, showfliers=False)
    if i % 2 == 0:
        fig.axes[i].set_ylabel('MSE')
    fig.axes[i].set_title(f[0][0],fontsize=labelsize)
    fig.axes[i].tick_params(axis="x", width=5, length=10)
    fig.axes[i].tick_params(axis="y", width=5, length=10)
    plt.setp(fig.axes[i].get_xticklabels(), rotation=20)
    plt.setp(fig.axes[i].get_yticklabels(), visible=True)
    for k, line in enumerate(bp_dict['means']):
        x, y = line.get_xydata()[1] # top of median line
        fig.axes[i].text(x+0.3, y, "{:.2e}".format(np.mean(boxplot_data[k])), horizontalalignment='right', rotation=-45)#, verticalalignment='bottom') # draw above, centered
        # fig.axes[i].set_yticks(10**(-9) * np.array([0.5,1,1.5]))

# fig.suptitle('expl_loss_min',size=50)
fig.tight_layout()
# plt.ylim([0, 10**(-9)*3])
# plt.yticks([0,10**(-9)*1, 10**(-9)*2])
# plt.yscale('linear')
# plt.gca().set_yscale([0,1,2,3])
plt.savefig('test_inception_imnet.png')
plt.show()
plt.close()



## expl training loss

import matplotlib.pyplot as plt

font_size = 25
labelsize = 32
axes_size = 30
BIGGER_SIZE = 20

plt.rc('font', size=font_size)          # controls default text sizes
plt.rc('axes', titlesize=axes_size)     # fontsize of the axes title
plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
plt.rc('legend', fontsize=38)



from collections import defaultdict
frame_dict = defaultdict(list)
frame_ = frame[['n_pop', 'method', 'lr', 'LS', 'input_loss', 'output_loss', 'expl_loss']]

for index, f in frame_.iterrows():
    title = f"n_pop={f['n_pop']}, lr={f['lr']}, LS={f['LS']}"
    frame_dict[(f['n_pop'], f['lr'], f['LS'])].append([(title, f['method'], f['expl_loss'])])
len(frame_dict.keys())
fig, axs = plt.subplots(4, 2, figsize=(27, 27), sharey='all')
for i, key in enumerate(frame_dict.keys()):
    boxplot_data = []
    boxplot_labels = []
    boxplot_label_order = []
    for _, f in enumerate(frame_dict[key]):
        data = np.array(f[0][2])
        bp_dict = fig.axes[i].plot(np.arange(data.shape[0]), data, label=f[0][1],linewidth=5, color='C'+str(attack_order.index(f[0][1])))
    fig.axes[i].tick_params(axis="x", width=5, length=10)
    fig.axes[i].tick_params(axis="y", width=5, length=10)
    fig.axes[i].set_title('        ' + f[0][0],fontsize=labelsize)
    if i % 2 == 0:
        fig.axes[i].set_ylabel('MSE')

import matplotlib.lines as mlines
lines_labels = fig.axes[0].get_legend_handles_labels()
lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
fig.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc="upper right")
fig.tight_layout()
plt.subplots_adjust(right=0.8)

plt.savefig('test_4.png')#,  bbox_inches="tight")
plt.show()
plt.close()






## input loss min
attack_order = choices=['lrp', 'guided_backprop', 'gradient',# 'integrated_grad',
# 'pattern_attribution', 
'grad_times_input', 'deep_lift']


from collections import defaultdict
frame_dict = defaultdict(list)
frame_ = frame[['n_pop', 'method', 'lr', 'LS', 'input_loss_min', 'output_loss_min', 'expl_loss_min']]

for index, f in frame_.iterrows():
    title = f"n_pop={f['n_pop']}, lr={f['lr']}, LS={f['LS']}"
    frame_dict[(f['n_pop'], f['lr'], f['LS'])].append([(title, f['method'], f['input_loss_min'])])

len(frame_dict.keys())

fig, axs = plt.subplots(4, 2,figsize=(27, 27))#, sharey='all')
# axs[-1,-1].set_axis_off()

for i, key in enumerate(frame_dict.keys()):
    boxplot_data = []
    boxplot_labels = []
    boxplot_label_order = []
    for _, f in enumerate(frame_dict[key]):
        boxplot_data.append(np.array(f[0][2]))
        boxplot_labels.append(f[0][1])
        boxplot_label_order.append(attack_order.index(f[0][1]))
    bp_dict = fig.axes[i].boxplot(boxplot_data, labels=boxplot_labels, positions=boxplot_label_order, meanline=True, showmeans=True, showfliers=False)
    # if i % 2 == 0:
        # fig.axes[i].set_ylabel('MSE')
    fig.axes[i].set_title(f[0][0],fontsize=labelsize)
    fig.axes[i].tick_params(axis="x", width=5, length=10)
    fig.axes[i].tick_params(axis="y", width=5, length=10)
    plt.setp(fig.axes[i].get_xticklabels(), rotation=20)
    plt.setp(fig.axes[i].get_yticklabels(), visible=True)
    for k, line in enumerate(bp_dict['means']):
        x, y = line.get_xydata()[1] # top of median line
        fig.axes[i].text(x+0.3, y, "{:.2e}".format(np.mean(boxplot_data[k])), horizontalalignment='right', rotation=-45)#, verticalalignment='bottom') # draw above, centered
        # fig.axes[i].set_yticks(10**(-9) * np.array([0.5,1,1.5]))

# fig.suptitle('expl_loss_min',size=50)
fig.tight_layout()
# plt.ylim([0, 10**(-9)*3])
# plt.yticks([0,10**(-9)*1, 10**(-9)*2])
# plt.yscale('linear')
# plt.gca().set_yscale([0,1,2,3])
plt.savefig('test.png')
plt.show()
plt.close()




## input training loss

import matplotlib.pyplot as plt

font_size = 25
labelsize = 32
axes_size = 30
BIGGER_SIZE = 20

plt.rc('font', size=font_size)          # controls default text sizes
plt.rc('axes', titlesize=axes_size)     # fontsize of the axes title
plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
plt.rc('legend', fontsize=38)



from collections import defaultdict
frame_dict = defaultdict(list)
frame_ = frame[['n_pop', 'method', 'lr', 'LS', 'input_loss', 'output_loss', 'expl_loss']]

for index, f in frame_.iterrows():
    title = f"n_pop={f['n_pop']}, lr={f['lr']}, LS={f['LS']}"
    frame_dict[(f['n_pop'], f['lr'], f['LS'])].append([(title, f['method'], f['expl_loss'])])
len(frame_dict.keys())
fig, axs = plt.subplots(4, 2, figsize=(27, 27), sharey='all')
for i, key in enumerate(frame_dict.keys()):
    boxplot_data = []
    boxplot_labels = []
    boxplot_label_order = []
    for _, f in enumerate(frame_dict[key]):
        data = np.array(f[0][2])
        bp_dict = fig.axes[i].plot(np.arange(data.shape[0]), data, label=f[0][1],linewidth=5, color='C'+str(attack_order.index(f[0][1])))
    fig.axes[i].tick_params(axis="x", width=5, length=10)
    fig.axes[i].tick_params(axis="y", width=5, length=10)
    fig.axes[i].set_title('        ' + f[0][0],fontsize=labelsize)
    if i % 2 == 0:
        fig.axes[i].set_ylabel('MSE')

import matplotlib.lines as mlines
lines_labels = fig.axes[0].get_legend_handles_labels()
lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
fig.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc="upper right")
fig.tight_layout()
plt.subplots_adjust(right=0.8)

plt.savefig('test_4.png')#,  bbox_inches="tight")
plt.show()
plt.close()






# cp -r /home/snirvit/AttaXAI/src/output /cs_storage/public_datasets/results/old_results_for_freeing_memory
# src = /home/snirvit/AttaXAI/src/argparser
# dest = /cs_storage/public_datasets/results
# cp -r /home/snirvit/AttaXAI/src/argparser /cs_storage/public_datasets/results
# cp -r /home/snirvit/AttaXAI/src/loss_file /cs_storage/public_datasets/results