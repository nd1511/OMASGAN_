import torch
import torch.nn as nn
import torchvision
import os
import utils
import losses
import numpy as np
import sys
from PIL import Image
import torchvision.transforms as transforms
def dummy_training_function():
    def train(x, y):
        return {}
    return train
def select_loss(config):
    if config['loss_type'] == 'hinge':
        return losses.loss_hinge_dis, losses.loss_hinge_gen
    elif 'kl' in config['loss_type']:
        temp = float(config['loss_type'].split('_')[-1])
        print('temp = ', temp)
        def dis_f(x, y):
            return losses.loss_kl_dis(x, y, temp)
        def gen_f(x):
            return losses.loss_kl_gen(x, temp)
        return dis_f, gen_f
    else:
        raise ValueError('loss not defined')
def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
    discriminator_loss, generator_loss = select_loss(config)
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)
        for step_index in range(config['num_D_steps']):
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                if not config['conditional']:
                    y_.zero_()
                    y_counter = torch.zeros_like(
                        y[counter]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y[counter]
                real_samples = x[counter]
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], real_samples,
                                    y_counter, train_G=False, split_D=config['split_D'])
                D_loss_real, D_loss_fake = discriminator_loss(
                    D_fake, D_real)
                D_loss = D_loss_real + D_loss_fake
                D_loss.backward()
                counter += 1
            if config['D_ortho'] > 0.0:
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])
            D.optim.step()
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)
        G.optim.zero_grad()
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            if not config['conditional']:
                y_.zero_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = generator_loss(
                D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()
        if config['G_ortho'] > 0.0:
            # Debug print to indicate we're using ortho reg in G
            print('using modified ortho reg in G')
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out
    return train


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (
            state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        if not config['conditional']:
            y_.zero_()
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(
                which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')


def update_FID(G, D, G_ema, state_dict, config, FID, experiment_name, test_log):
    print('Itr %d: PYTORCH UNOFFICIAL FID is %5.4f' %
          (state_dict['itr'], FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (
            state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(0),
                 IS_std=float(0), FID=float(FID))
