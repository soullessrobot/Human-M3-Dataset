# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models import pose_resnet
from models.cuboid_proposal_net import CuboidProposalNet
from models.pose_regression_net import PoseRegressionNet
from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss
from models.project_layer import ProjectLayer

class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.cfg = cfg
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.root_net = CuboidProposalNet(cfg)
        self.pose_net = PoseRegressionNet(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET

    def forward(self, dict_):
        # views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None
        views = dict_['input'] if 'input' in dict_ else None
        meta = dict_['meta'] if 'meta' in dict_ else None
        targets_2d = dict_['target_2d'] if 'target_2d' in dict_ else None
        weights_2d = dict_['weight_2d'] if 'weight_2d' in dict_ else None
        targets_3d = dict_['target_3d'][0] if 'target_3d' in dict_ else None
        input_heatmaps = dict_['input_heatmap'] if 'input_heatmap' in dict_ else None
        pcd_voxel = dict_['pcd_voxel'] if 'pcd_voxel' in dict_ else None

        if views is not None and self.backbone is not None:
            all_heatmaps = []
            for view in views:
                feature = self.backbone(view)
                all_heatmaps.append(feature)
        else:
            all_heatmaps = input_heatmaps
            
        device = views[0].device if views is not None else pcd_voxel.device
        batch_size = views[0].shape[0] if views is not None else pcd_voxel.shape[0]
        criterion = PerJointMSELoss().cuda()
        loss_2d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        loss_3d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        if self.USE_GT:
            num_person = meta[0]['num_person']
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, :num_person[i], 4] = 1.0
        else:
            root_cubes, grid_centers = self.root_net(all_heatmaps, meta, pcd_voxel)

            # calculate 3D heatmap loss
            if targets_3d is not None:
                loss_3d = criterion(root_cubes, targets_3d)
            del root_cubes
            
        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt

        loss_cord = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        criterion_cord = PerJointL1Loss().cuda()
        count = 0

        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:
                single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], pcd_voxel)
                pred[:, n, :, 0:3] = single_pose.detach()
                if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                    gt_3d = meta[0]['joints_3d'].float()
                    for i in range(batch_size):
                        if pred[i, n, 0, 3] >= 0:
                            targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()]
                            weights_3d = meta[0]['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                            count += 1
                            loss_cord = (loss_cord * (count - 1) +
                                         criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)) / count
                del single_pose
        del dict_
        return pred, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord

def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        # 
        if cfg.NETWORK.INPUT_HM:
            backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
        else:
            backbone = nn.Sequential(*list(eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train).children()))
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
