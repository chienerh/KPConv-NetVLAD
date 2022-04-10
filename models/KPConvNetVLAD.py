
"""
network architechture for place recognition (Oxford dataset)

"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from models.architectures import KPConvEncoder
from models.NetVLADLoupe import NetVLADLoupe
import config as cfg
from utils.config import Config
from datasets.common import *

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class OxfordConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'Oxford'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'place recognition'

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    # architecture = ['simple',
    #                'resnetb',
    #                'resnetb',
    #                'resnetb_strided',
    #                'resnetb',
    #                'resnetb',]

    architecture = ['simple',
                   'resnetb',
                   'resnetb']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02 #0.02, 0.03

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 1

    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.05

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1**(1/100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 22

    # Number of steps per epochs
    epoch_steps = 300

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None


class KPConvData:
    def __init__(self, input_points, input_neighbors, input_pools, input_stack_lengths, stacked_features):
        self.points = input_points
        self.neighbors = input_neighbors
        self.pools = input_pools
        self.stack_lengths = input_stack_lengths
        self.features = torch.tensor(stacked_features, device=input_points[0].device)


def preprocess_input(dataset_input, config):
    """
    Preprocess input point clouds for KPConv
    dataset_input: BxNx3
    Code taken from https://github.com/yewzijian/RegTR/blob/64e5b3f0ccc1e1a11b514eb22734959d32e0cec6/src/models/backbone_kpconv/kpconv.py#L417
    and https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/5b5641e02daac0043adfe97724de8c771dd4772f/datasets/common.py#L205
    """
    neighborhood_limits = [50, 50]

    def big_neighborhood_filter(neighbors, layer, neighborhood_limits):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(neighborhood_limits) > 0:
            return neighbors[:, :neighborhood_limits[layer]]
        else:
            return neighbors


    # Input features
    device = dataset_input.device
    stacked_points = np.array(dataset_input.detach().cpu().reshape(-1,3))
    stack_lengths = np.array([tp.shape[0] for tp in stacked_points], dtype=np.int32)
    stacked_features = np.ones_like(stacked_points[:, :1])

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_stack_lengths = []
    deform_layers = []

    ######################
    # Loop over the blocks
    ######################

    arch = config.architecture

    for block_i, block in enumerate(arch):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(arch)-1 and not ('upsample' in arch[block_i+1]):
                continue

        # Convolution neighbors indices
        # *****************************

        deform_layer = False
        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
                deform_layer = True
            else:
                r = r_normal

            conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)
            if np.max(conv_i) > stacked_points.shape[0]:
                convi[np.unravel_index(convi.argmax(), convi.shape)] -= 1
                print('changed neighbor')

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = np.zeros((0, 1), dtype=np.int32)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
                deform_layer = True
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            # print('block_i', block_i, 'stacked_points', stacked_points.shape, 'max neighbor', np.max(conv_i), 'max pool', np.max(pool_i))


        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = np.zeros((0, 1), dtype=np.int32)
            pool_p = np.zeros((0, 1), dtype=np.float32)
            pool_b = np.zeros((0,), dtype=np.int32)
            # print('block_i', block_i, 'stacked_points', stacked_points.shape, 'max neighbor', np.max(conv_i))


        # Reduce size of neighbors matrices by eliminating furthest point
        conv_i = big_neighborhood_filter(conv_i, len(input_points), neighborhood_limits)
        pool_i = big_neighborhood_filter(pool_i, len(input_points), neighborhood_limits)
        # Updating input lists
        input_points += [torch.tensor(stacked_points, device=device)]
        input_neighbors += [torch.tensor(conv_i.astype(np.int64), device=device)]
        input_pools += [torch.tensor(pool_i.astype(np.int64), device=device)]
        input_stack_lengths += [torch.tensor(stack_lengths, device=device)]
        deform_layers += [deform_layer]

        # New points for next layer
        stacked_points = pool_p
        stack_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer_blocks = []


    ###############
    # Return inputs
    ###############

    # Save deform layers

    # list of network inputs
    # li = input_points + input_neighbors + input_pools + input_stack_lengths
    # li += [stacked_features]

    data_output = KPConvData(input_points, input_neighbors, input_pools, input_stack_lengths, stacked_features)

    return data_output


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

class KPConvNetVLAD(nn.Module):
    def __init__(self, config=None):
        super(KPConvNetVLAD, self).__init__()
        if config is None:
            self.config = OxfordConfig()
        else:
            self.config = config
        
        self.kpconv = KPConvEncoder(self.config)

        print('KPConv Network\n', self.kpconv )

        self.netvlad = NetVLADLoupe(feature_size=cfg.LOCAL_FEATURE_DIM, max_samples=cfg.NUM_POINTS, cluster_size=64,
                                     output_dim=cfg.FEATURE_OUTPUT_DIM, gating=True, add_batch_norm=True,
                                     is_training=True)

        print('Finished Model Initialization')

    def forward(self, x):
        '''
        INPUT: B, N, D_input=3
        Local Feature: B, N', D_local
        Global Feature: B, D_output
        '''
        B, Q, N, _ = x.shape
        batch = preprocess_input(x, self.config)

        x = self.kpconv(batch)
        # print('x after kpconv', x.shape)
        x = x.view(B, N, cfg.LOCAL_FEATURE_DIM)
        x_frontend = x
        # print('x_frontend', x_frontend.shape)

        x = self.netvlad(x)

        return x, x_frontend

