from __future__ import print_function
# Use: --select_dataset cifar10 --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
# Usage: --select_dataset mnist --abnormal_class 1 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
# Use: --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
# Acknowledgements: Thanks to the repositories: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#mu_select = mu_select
#mu_select = 0.2
#ni_select = ni_select
#ni_select = 0.3
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import inception_utils
import utils
import losses
import train_fns
import fid_score
from sync_batchnorm import patch_replication_callback
import sys
def run(config):
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    if config['resume']:
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'
    utils.seed_rng(config['seed'])
    utils.prepare_root(config)
    torch.backends.cudnn.benchmark = True
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)
    if config['ema']:
        G_ema = model.Generator(**{**config, 'skip_init': True, 'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None
    if config['G_fp16']:
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        D = D.half()
    GD = model.G_D(G, D, config['conditional'])
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_IS': 0, 'best_FID': 999999, 'config': config}
    if config['resume']:
        utils.load_weights(G, D, state_dict, config['weights_root'], experiment_name, config['load_weights'] if config['load_weights'] else None, G_ema if config['ema'] else None)
    utils.load_weights(G, D, state_dict, '../Task1_CIFAR_MNIST_KLWGAN_Simulation_Experiment/weights', 'C10Ukl5', 'best0', G_ema if config['ema'] else None)
    if config['parallel']:
        GD = nn.DataParallel(GD)
        if config['cross_replica']:
            patch_replication_callback(GD)
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'], experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    test_log = utils.MetricsLogger(test_metrics_fname, reinitialize=(not config['resume']))
    train_log = utils.MyLogger(train_metrics_fname, reinitialize=(not config['resume']), logstyle=config['logstyle'])
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
    # Use: config['abnormal_class']
    #print(config['abnormal_class'])
    #abnormal_class = config['abnormal_class']
    #select_dataset = config['select_dataset']
    #print(config['select_dataset'])
    #print(abnormal_class)
    #print(select_dataset)
    abnormal_class = config['abnormal_class']
    select_dataset = config['select_dataset']
    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size, 'start_itr': state_dict['itr'], 'abnormal_class': abnormal_class, 'select_dataset': select_dataset})
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    if not config['conditional']:
        fixed_y.zero_()
        y_.zero_()
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config)
    else:
        train = train_fns.dummy_training_function()
    sample = functools.partial(utils.sample, G=(G_ema if config['ema'] and config['use_ema'] else G), z_=z_, y_=y_, config=config)
    if config['dataset'] == 'C10U' or config['dataset'] == 'C10':
        #data_moments = 'fid_stats_cifar10_train.npz'
        #'../Task1_CIFAR_MNIST_KLWGAN_Simulation_Experiment/fid_stats_cifar10_train.npz'
        data_moments = '../Task1_CIFAR_MNIST_KLWGAN_Simulation_Experiment/fid_stats_cifar10_train.npz'
    else:
        print("Cannot find the image data set.")
        sys.exit()
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        if config['pbar'] == 'mine':
            pbar = utils.progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        for i, (x, y) in enumerate(pbar):
            state_dict['itr'] += 1
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            print('')
            # Random seed
            #print(config['seed'])
            if epoch==0 and i==0:
                print(config['seed'])
            # We double the learning rate if we double the batch size.
            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']), **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']] + ['%s : %+4.3f' % (key, metrics[key]) for key in metrics]), end=' ')
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name)
            experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
            if (not (state_dict['itr'] % config['test_every'])) and (epoch >= config['start_eval']):
                if config['G_eval_mode']:
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                utils.sample_inception(G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                folder_number = str(epoch)
                sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                FID = fid_score.calculate_fid_given_paths([data_moments, sample_moments], batch_size=50, cuda=True, dims=2048)
                train_fns.update_FID(G, D, G_ema, state_dict, config, FID, experiment_name, test_log)
        state_dict['epoch'] += 1
    utils.save_weights(G, D, state_dict, config['weights_root'], experiment_name, 'last%d' % 0, G_ema if config['ema'] else None)
    # Save the last model
    #utils.save_weights(G, D, state_dict, config['weights_root'], experiment_name, 'last%d' % 0, G_ema if config['ema'] else None)
# We create dynamics by pushing the generated samples OoD: Likelihood-free boundary of data distribution
# The first two terms of the proposed objective cost function, i.e. $- m( B, G)$ and $d( B, G )$,
# are convex as functions of the underlying probability measures. Because of the neural architectures
# used, the optimization problem is non-convex. Regarding the choice of loss function for the boundary
# Task, we create dynamics by pushing the generated samples OoD. Taking into account results from
# optimization theory and optimization results for non-linear architectures, we introduce a regularized
# cost function in the optimization instead of attempting to solve a much harder constrained optimization.
def main():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    # Print the configuration.
    #print(config)
    run(config)
if __name__ == '__main__':
    main()
# The use of torch.nn.DataParallel(model) is recommended along
# with the use of torch.save(model.module.state_dict(), "./.pt") instead
# of torch.save(model.state_dict(), "./.pt"). Also, saving the best model
# is recommended by using "best_loss = float('inf')" and "if loss.item()<best_loss:
# best_loss=loss.item(); torch.save(model.module.state_dict(), "./.pt")". Downloading the
# image dataset one time is also recommended, e.g. "--data_root ../<path-to-folder-of-dataset>/data/".
# For the evaluation of OMASGAN on MNIST image data, we obtain
# (https://github.com/Anonymous-Author-2021/OMASGAN/blob/main/Simulations_Experiments/Images_Generated_MNIST_Task3_fGAN.pdf)
# and for the evaluation of OMASGAN on CIFAR-10 data, we obtain
# (https://github.com/Anonymous-Author-2021/OMASGAN/blob/main/Simulations_Experiments/Images_Generated_CIFAR-10_Task3_KLWGAN.pdf).
# Acknowledgements: Thanks to the repositories: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).
# Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).
# Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).
# All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".
