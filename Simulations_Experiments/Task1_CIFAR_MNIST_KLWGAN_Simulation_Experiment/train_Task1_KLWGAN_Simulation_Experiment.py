from __future__ import print_function
# Use: --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
# Use: --abnormal_class 1 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
# Use: python train_Task1_KLWGAN_Proof_of_Concept.py --abnormal_class 1 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
# Acknowledgement: Thanks to the repository: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Implicit generative models and GANs generate sharp, low-FID, realistic, and high-quality images.
# We use implicit generative models and GANs for the challenging task of anomaly detection in high-dimensional spaces.
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
import utils_Task1_KLWGAN_Simulation_Experiment
from utils_Task1_KLWGAN_Simulation_Experiment import *
import losses_Task1_KLWGAN_Simulation_Experiment
import train_fns
import fid_score
import sys
from sync_batchnorm import patch_replication_callback
# Choose and select dataset.
#select_dataset = "select_dataset"
select_dataset = "cifar10"
#select_dataset = "mnist"
def run(config):
    config['resolution'] = imsize_dict[config['dataset']]
    config['n_classes'] = nclass_dict[config['dataset']]
    config['G_activation'] = activation_dict[config['G_nl']]
    config['D_activation'] = activation_dict[config['D_nl']]
    if config['resume']:
        config['skip_init'] = True
    config = update_config_roots(config)
    device = 'cuda'
    utils_Task1_KLWGAN_Simulation_Experiment.seed_rng(config['seed'])
    utils_Task1_KLWGAN_Simulation_Experiment.prepare_root(config)
    torch.backends.cudnn.benchmark = True
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils_Task1_KLWGAN_Simulation_Experiment.name_from_config(config))
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)
    if config['ema']:
        G_ema = model.Generator(**{**config, 'skip_init': True, 'no_optim': True}).to(device)
        ema = utils_Task1_KLWGAN_Simulation_Experiment.ema(G, G_ema, config['ema_decay'], config['ema_start'])
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
        utils_Task1_KLWGAN_Simulation_Experiment.load_weights(G, D, state_dict, config['weights_root'], experiment_name, config['load_weights'] if config['load_weights'] else None, G_ema if config['ema'] else None)
    if config['parallel']:
        GD = nn.DataParallel(GD)
        if config['cross_replica']:
            patch_replication_callback(GD)
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'], experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    test_log = utils_Task1_KLWGAN_Simulation_Experiment.MetricsLogger(test_metrics_fname, reinitialize=(not config['resume']))
    train_log = utils_Task1_KLWGAN_Simulation_Experiment.MyLogger(train_metrics_fname, reinitialize=(not config['resume']), logstyle=config['logstyle'])
    utils_Task1_KLWGAN_Simulation_Experiment.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
    # Use: config['abnormal_class']
    #print(config['abnormal_class'])
    abnormal_class = config['abnormal_class']
    loaders = utils_Task1_KLWGAN_Simulation_Experiment.get_data_loaders(**{**config, 'batch_size': D_batch_size, 'start_itr': state_dict['itr'], 'abnormal_class': abnormal_class})
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils_Task1_KLWGAN_Simulation_Experiment.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z, fixed_y = utils_Task1_KLWGAN_Simulation_Experiment.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    if not config['conditional']:
        fixed_y.zero_()
        y_.zero_()
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config)
    else:
        train = train_fns.dummy_training_function()
    sample = functools.partial(utils_Task1_KLWGAN_Simulation_Experiment.sample, G=(G_ema if config['ema'] and config['use_ema'] else G), z_=z_, y_=y_, config=config)
    if config['dataset'] == 'C10U' or config['dataset'] == 'C10':
        data_moments = 'fid_stats_cifar10_train.npz'
    else:
        print("Cannot find the dataset.")
        sys.exit()
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        if config['pbar'] == 'mine':
            pbar = utils_Task1_KLWGAN_Simulation_Experiment.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
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
            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']), **{**utils_Task1_KLWGAN_Simulation_Experiment.get_SVs(G, 'G'), **utils_Task1_KLWGAN_Simulation_Experiment.get_SVs(D, 'D')})
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']] + ['%s : %+4.3f' % (key, metrics[key]) for key in metrics]), end=' ')
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name)
            experiment_name = (config['experiment_name'] if config['experiment_name'] else utils_Task1_KLWGAN_Simulation_Experiment.name_from_config(config))
            if (not (state_dict['itr'] % config['test_every'])) and (epoch >= config['start_eval']):
                if config['G_eval_mode']:
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                utils_Task1_KLWGAN_Simulation_Experiment.sample_inception(
                    G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                folder_number = str(epoch)
                sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                #FID = fid_score.calculate_fid_given_paths([data_moments, sample_moments], batch_size=50, cuda=True, dims=2048)
                #train_fns.update_FID(G, D, G_ema, state_dict, config, FID, experiment_name, test_log)
                # Use the files train_fns.py and utils_Task1_KLWGAN_Simulation_Experiment.py
                # Use the functions update_FID() and save_weights()
                # Save the lowest FID score
                FID = fid_score.calculate_fid_given_paths([data_moments, sample_moments], batch_size=50, cuda=True, dims=2048)
                train_fns.update_FID(G, D, G_ema, state_dict, config, FID, experiment_name, test_log)
        state_dict['epoch'] += 1
    # Save the last model
    utils_Task1_KLWGAN_Simulation_Experiment.save_weights(G, D, state_dict, config['weights_root'], experiment_name, 'last%d' % 0, G_ema if config['ema'] else None)
def main():
    parser = utils_Task1_KLWGAN_Simulation_Experiment.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)
if __name__ == '__main__':
    main()
# Acknowledgement: Thanks to the repository: [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)
# Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
