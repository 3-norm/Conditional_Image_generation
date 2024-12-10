''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



# Arguments for sample.py; not presently used in train.py
def add_sample_parser(parser):
  parser.add_argument(
    '--sample_npz', action='store_true', default=False,
    help='Sample "sample_num_npz" images and save to npz? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_num_npz', type=int, default=50000,
    help='Number of images to sample when sampling NPZs '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_sheets', action='store_true', default=False,
    help='Produce class-conditional sample sheets and stick them in '
         'the samples root? (default: %(default)s)')
  # parser.add_argument(
  #   '--sample_interps', action='store_true', default=False,
  #   help='Produce interpolation sheets and stick them in '
  #        'the samples root? (default: %(default)s)')         
  parser.add_argument(
    '--sample_sheet_folder_num', type=int, default=-1,
    help='Number to use for the folder for these sample sheets '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_random', action='store_true', default=False,
    help='Produce a single random sheet? (default: %(default)s)')
  parser.add_argument(
    '--sample_trunc_curves', type=str, default='',
    help='Get inception metrics with a range of variances?'
         'To use this, specify a startpoint, step, and endpoint, e.g. '
         '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
         'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
         'not exactly identical to using tf.truncated_normal, but should '
         'have approximately the same effect. (default: %(default)s)')
  # parser.add_argument(
  #   '--sample_inception_metrics', action='store_true', default=False,
  #   help='Calculate Inception metrics with sample.py? (default: %(default)s)')  
  return parser


    
# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler(torch.utils.data.Sampler):
  r"""Samples elements randomly over multiple epochs

  Arguments:
      data_source (Dataset): dataset to sample from
      num_epochs (int) : Number of times to loop over the dataset
      start_itr (int) : which iteration to begin from
  """

  def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
    self.data_source = data_source
    self.num_samples = len(self.data_source)
    self.num_epochs = num_epochs
    self.start_itr = start_itr
    self.batch_size = batch_size

    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
      raise ValueError("num_samples should be a positive integeral "
                       "value, but got num_samples={}".format(self.num_samples))

  def __iter__(self):
    n = len(self.data_source)
    # Determine number of epochs
    num_epochs = int(np.ceil((n * self.num_epochs 
                              - (self.start_itr * self.batch_size)) / float(n)))
    # Sample all the indices, and then grab the last num_epochs index sets;
    # This ensures if we're starting at epoch 4, we're still grabbing epoch 4's
    # indices
    out = [torch.randperm(n) for epoch in range(self.num_epochs)][-num_epochs:]
    # Ignore the first start_itr % n indices of the first epoch
    out[0] = out[0][(self.start_itr * self.batch_size % n):]
    # if self.replacement:
      # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
    # return iter(.tolist())
    output = torch.cat(out).tolist()
    print('Length dataset output is %d' % len(output))
    return iter(output)

  def __len__(self):
    return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size




# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
  if config['base_root']:
    print('Pegging all root folders to base root %s' % config['base_root'])
    for key in ['data', 'weights', 'logs', 'samples']:
      config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
  return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
  for key in ['weights_root', 'logs_root', 'samples_root']:
    if not os.path.exists(config[key]):
      print('Making directory %s for %s...' % (config[key], key))
      os.mkdir(config[key])


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    # Initialize target's params to be source's
    self.source_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)
        # target_dict[key].data = source_dict[key].data # Doesn't work!

  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay 
                                     + self.source_dict[key].data * (1 - decay))


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes & not in blacklist
      if len(param.shape) < 2 or param in blacklist:
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
               - torch.eye(w.shape[0], device=w.device), w))
      param.grad.data += strength * grad.view(param.shape)


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])


# Save a model's weights, optimizer, and the state_dict
def save_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None):
  root = '/'.join([weights_root, experiment_name])
  if not os.path.exists(root):
    os.mkdir(root)
  if name_suffix:
    print('Saving weights to %s/%s...' % (root, name_suffix))
  else:
    print('Saving weights to %s...' % root)
  torch.save(G.state_dict(), 
              '%s/%s.pth' % (root, join_strings('_', ['G', name_suffix])))
  torch.save(G.optim.state_dict(), 
              '%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix])))
  torch.save(D.state_dict(), 
              '%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])))
  torch.save(D.optim.state_dict(),
              '%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix])))
  torch.save(state_dict,
              '%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))
  if G_ema is not None:
    torch.save(G_ema.state_dict(), 
                '%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix])))


# Load a model's weights, optimizer, and the state_dict
def load_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None, strict=True, load_optim=True):
  root = '/'.join([weights_root, experiment_name])
  if name_suffix:
    print('Loading %s weights from %s...' % (name_suffix, root))
  else:
    print('Loading weights from %s...' % root)
  if G is not None:
    G.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
      strict=strict)
    if load_optim:
      G.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
  if D is not None:
    D.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
      strict=strict)
    if load_optim:
      D.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
  # Load state dict
  for item in state_dict:
    state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]
  if G_ema is not None:
    G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
      strict=strict)



# Sample function for use with inception metrics
def sample(G, z_, y_, config):
  with torch.no_grad():
    z_.sample_()
    # y_.sample_()
    if config['parallel']:
      G_z =  nn.parallel.data_parallel(G, (z_, G.shared(y_)))
    else:
      G_z = G(z_, G.shared(y_))
    return G_z, y_


# Sample function for sample sheets
def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None):
  # Prepare sample directory
  if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
    os.mkdir('%s/%s' % (samples_root, experiment_name))
  if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
    os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))
  # loop over total number of sheets
  for i in range(num_classes // classes_per_sheet):
    ims = []
    y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
    for j in range(samples_per_class):
      if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
        z_.sample_()
      else:
        z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')        
      with torch.no_grad():
          o = G(z_[:classes_per_sheet], G.shared(y))
      
      ims += [o.data.cpu()]
    # This line should properly unroll the images
    out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2], 
                                       ims[0].shape[3]).data.float().cpu()
    out_ims = torch.from_numpy(out_ims.numpy())
    # The path for the samples
    image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name, 
                                                 folder_number, i)

    torchvision.utils.save_image(out_ims, image_filename,
                                 nrow=samples_per_class, normalize=True)



# Get singular values to log. This will use the state dict to find them
# and substitute underscores for dots.
def get_SVs(net, prefix):
  d = net.state_dict()
  return {('%s_%s' % (prefix, key)).replace('.', '_') :
            float(d[key].item())
            for key in d if 'sv' in key}


# Get GPU memory, -i is the index
def query_gpu(indices):
  os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')


# Convenience function to count the number of parameters in a module
def count_parameters(module):
  print('Number of parameters: {}'.format(
    sum([p.data.nelement() for p in module.parameters()])))

   
# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
  return torch.randint(low=0, high=num_classes, size=(batch_size,),
          device=device, dtype=torch.int64, requires_grad=False)


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)    
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', 
                fp16=False,z_var=1.0):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
  
  if fp16:
    z_ = z_.half()

  y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical',num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_


def initiate_standing_stats(net):
  for module in net.modules():
    if hasattr(module, 'accumulate_standing'):
      module.reset_stats()
      module.accumulate_standing = True


def accumulate_standing_stats(net, z, y, nclasses, num_accumulations=16):
  initiate_standing_stats(net)
  net.train()
  for i in range(num_accumulations):
    with torch.no_grad():
      z.normal_()
      y.random_(0, nclasses)
      x = net(z, net.shared(y)) # No need to parallelize here unless using syncbn
  # Set to eval mode
  net.eval() 


# This version of Adam keeps an fp32 copy of the parameters and
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params.
import math
from torch.optim.optimizer import Optimizer
class Adam16(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay)
    params = list(params)
    super(Adam16, self).__init__(params, defaults)
      
  # Safety modification to make sure we floatify our state
  def load_state_dict(self, state_dict):
    super(Adam16, self).load_state_dict(state_dict)
    for group in self.param_groups:
      for p in group['params']:
        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
        self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
          
        grad = p.grad.data.float()
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = grad.new().resize_as_(grad).zero_()
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
          # Fp32 copy of the weights
          state['fp32_p'] = p.data.float()

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], state['fp32_p'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
      
        state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
        p.data = state['fp32_p'].half()

    return loss
