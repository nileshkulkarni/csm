"""
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags
from ..utils.visualizer import Visualizer

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')
flags.DEFINE_bool('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as mmomentum')

flags.DEFINE_integer('batch_size', 7, 'Size of minibatches')
flags.DEFINE_integer('num_iter', 0, 'Number of training iterations. 0 -> Use epoch_iter')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')


## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 1000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 40, 'save model every k epochs')

## Flags for visualization
flags.DEFINE_integer('display_freq', 20, 'visuals logging frequency')
flags.DEFINE_boolean('display_visuals', True, 'whether to display images')

flags.DEFINE_integer('save_visual_freq', 1000, 'visuals save frequency')
flags.DEFINE_integer('save_visual_count', 6, 'visuals save count')
flags.DEFINE_boolean('save_visuals', True, 'whether to save images')

flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
flags.DEFINE_boolean('plot_scalars', True, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_boolean('clip_grad', False, 'Clip gradients')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')
flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('use_gt_quat', False, 'Use only the GT orientation. Only valid if used with pred_cam')
flags.DEFINE_integer('lr_step_epoch_freq', -1, 'Reduce LR by factor of 10 every k ephochs')

#-------- tranining class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        self.device = 'cuda:{}'.format(opts.gpu_id)
        self.cam_location = [0, 0, -2.732]
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))


    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda(device=gpu_id)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        print('Restoring weights from {}'.format(save_path))
        return

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix, gpu_id=self.opts.gpu_id)
        return

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        if opts.lr_step_epoch_freq < 0:
            opts.lr_step_epoch_freq = opts.num_epochs
        if opts.use_sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=opts.learning_rate, momentum=opts.beta1)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=opts.learning_rate, betas=(opts.beta1, 0.999))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opts.lr_step_epoch_freq, gamma=0.8)
    

    def compute_grad_norm(self, parameters):
        total_norm = 0
        norm_type = 2
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        # import ipdb ; ipdb.set_trace()
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm
    
    def train(self):
        opts = self.opts
        self.smoothed_total_loss = 0
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        self.total_steps = total_steps = 1
        self.real_iter = 0
        dataset_size = len(self.dataloader)
        start_time = time.time()

        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
            epoch_iter = 0
            self.scheduler.step()
            for i, batch in enumerate(self.dataloader):
                iter_start_time = time.time()
                self.set_input(batch)
                if not self.invalid_batch:
                    self.real_iter +=1
                    self.optimizer.zero_grad()
                    self.forward()
                    self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.total_loss.item()
                    self.total_loss.backward()
                    self.grad_norm = self.compute_grad_norm(self.model.parameters())
                    if self.grad_norm != self.grad_norm:
                        print('Ignoring batch')
                    self.optimizer.step()

                total_steps += 1
                epoch_iter += 1

                if opts.display_visuals and (total_steps % opts.display_freq == 0):
                    iter_end_time = time.time()
                    # print('time/itr %.2g' % ((iter_end_time - iter_start_time)/opts.display_freq))
                    visualizer.display_current_results(self.get_current_visuals(), epoch)
                    visualizer.plot_current_points(self.get_current_points())

                if opts.save_visuals and (total_steps % opts.save_visual_freq == 0):
                    visualizer.save_current_results(total_steps, self.visuals_to_save(total_steps))

                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    time_diff = time.time() - start_time
                    visualizer.print_current_scalars(time_diff, epoch, epoch_iter, scalars)
                    if opts.plot_scalars:
                        visualizer.plot_current_scalars(epoch, float(epoch_iter)/dataset_size, opts, scalars)


                if total_steps % opts.save_latest_freq == 0:
                    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return
                self.total_steps = total_steps
            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)
