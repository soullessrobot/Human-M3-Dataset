# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()
        self.cfg = cfg
        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER
        self.use_hm = cfg.NETWORK.INPUT_HM or cfg.NETWORK.USE_RGB
        self.use_pcd = cfg.NETWORK.USE_PCD
        self.project_type = cfg.NETWORK.PROJECT_TYPE
        assert (self.use_hm or self.use_pcd)

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]
        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps)
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, n, device=device)
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, nbins, n, device=device)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid
                for c in range(n):
                    width, height = meta[c]['image_size'][0][i], meta[c]['image_size'][1][i]
                    cam = {}
                    for k, v in meta[c]['camera'].items():
                        cam[k] = v[i]
                    xy = cameras.project_pose(grid, cam)
                    bounding[i, 0, 0, :, c] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                                xy[:, 1] < height)
                    xy = torch.clamp(xy, torch.tensor(-1.0).cuda(), max(width, height))
                    sample_grid = xy / torch.tensor(
                        [width - 1, height - 1], dtype=torch.float,
                        device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)
                    cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True)

        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-6)
        cubes[cubes != cubes] = 0.0
        if self.project_type == 'heatmap':
            cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_joints, cube_size[0], cube_size[1], cube_size[2])  ##
        return cubes, grids

    def compute_grid_voxel(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]
        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def get_pcd_voxel(self, pcd_voxel, grid_size, grid_center, cube_size):
        device = pcd_voxel.device
        batch_size = pcd_voxel.shape[0]
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        cubes = torch.zeros(batch_size, 1, cube_size[0], cube_size[1], cube_size[2], device=device)
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid #[N,3]
                grid = grid.view(1, cube_size[0], cube_size[1], cube_size[2], 3)
                sample_grid = grid / torch.tensor(self.cfg.MULTI_PERSON.SPACE_SIZE, device = device).unsqueeze(0) * 2 - 1
                sample_grid = torch.clamp(sample_grid, -1.1, 1.1)
                sample_grid[...,[0,2]] = sample_grid[...,[2,0]]
                cubes[i:i + 1, ...] += F.grid_sample(pcd_voxel[i:i+1], sample_grid, align_corners=True, mode = 'bilinear')
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, 1, cube_size[0], cube_size[1], cube_size[2])  ##
        return cubes, grids

    def forward(self, heatmaps = None, meta = None, grid_size = None, grid_center = None, cube_size = None, pcd_voxel = None):
        if self.use_hm and heatmaps is not None:
            cubes, grids = self.get_voxel(heatmaps, meta, grid_size, grid_center, cube_size)
        if self.use_pcd and pcd_voxel is not None:
            cubes_voxel, grids_voxel = self.get_pcd_voxel(pcd_voxel, grid_size, grid_center, cube_size)
            cubes = torch.cat([cubes, cubes_voxel], dim = 1) if (self.use_hm and heatmaps is not None) else cubes_voxel
            grids = grids_voxel if not (self.use_hm and heatmaps is not None) else grids
        return cubes, grids
