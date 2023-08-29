# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        # import pdb; pdb.set_trace()
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x

class PoseRegressionNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.project_layer = ProjectLayer(cfg)
        if cfg.NETWORK.FUSION_TYPE == 'concat_hm':
            n_input = cfg.NETWORK.NUM_JOINTS * cfg.NETWORK.INPUT_HM + cfg.NETWORK.USE_PCD \
             if cfg.NETWORK.INPUT_HM else cfg.NETWORK.FEAT_CHANNEL * cfg.NETWORK.USE_RGB + cfg.NETWORK.USE_PCD
            self.v2v_net = V2VNet(n_input, cfg.NETWORK.NUM_JOINTS)
        elif cfg.NETWORK.FUSION_TYPE == 'mask':
            n_input = cfg.NETWORK.NUM_JOINTS if cfg.NETWORK.INPUT_HM else cfg.NETWORK.FEAT_CHANNEL
            self.v2v_net = V2VNet(n_input, cfg.NETWORK.NUM_JOINTS)
        elif cfg.NETWORK.FUSION_TYPE == 'trans':
            n_joints = cfg.NETWORK.NUM_JOINTS
            n_pcd = cfg.NETWORK.USE_PCD + 0
            self.v2v_net_joints = V2VNet(n_joints, 64)
            self.v2v_net_pcd = V2VNet(n_pcd, 64)
            self.fusion_module = nn.Conv3d(128, cfg.NETWORK.NUM_JOINTS, kernel_size = 1, stride = 1)
            
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)
        self.cfg = cfg

    def forward(self, all_heatmaps = None, meta = None, grid_centers = None, pcd_voxel = None):
        batch_size = all_heatmaps[0].shape[0] if all_heatmaps is not None else pcd_voxel.shape[0]
        num_joints = self.num_joints #all_heatmaps[0].shape[1] if all_heatmaps is not None else pcd_voxel.shape[1]
        device = all_heatmaps[0].device if all_heatmaps is not None else pcd_voxel.device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        if grid_centers is not None:
            index = grid_centers[:, 3] >= 0
            cubes, grids = self.project_layer(all_heatmaps, meta,
                                            self.grid_size, grid_centers, self.cube_size, pcd_voxel)
            valid_cubes = self.v2v_net(cubes[index])
            pred[index] = self.soft_argmax_layer(valid_cubes, grids[index])
        else:
            grid_centers = torch.zeros(batch_size, 3).to(pcd_voxel.device)
            cubes, grids = self.project_layer(all_heatmaps, meta,
                                            self.grid_size, grid_centers, self.cube_size, pcd_voxel)
            valid_cubes = self.v2v_net(pcd_voxel)
            pred = self.soft_argmax_layer(valid_cubes, grids)
        return pred
