import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models import NECKS

from .mapping_table_v2 import MappingTableV2


class CHAttention(nn.Module):
    def __init__(self, in_planes):
        super(CHAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return residual * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return residual * self.sigmoid(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()

        self.cha = CHAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.cha(x)
        x = self.sa(x)

        return x


@NECKS.register_module()
class ConvOccLSVT(BaseModule):
    def __init__(self,
                 n_voxels,
                 pc_range,
                 in_channels,
                 out_channels,
                 use_height_attention,

                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 ):
        super(ConvOccLSVT, self).__init__()

        self.n_voxels = n_voxels
        self.pc_range = pc_range
        self._init_points(pc_range, n_voxels)

        self.upsample_cfg = upsample_cfg

        # 高度注意力
        self.use_height_attention = use_height_attention
        if self.use_height_attention:
            self.attentions = nn.ModuleList()
            for i in range(len(n_voxels)):
                self.attentions.append(BasicBlock(in_channels))

        # 多尺度融合
        self.lateral_convs = nn.ModuleList()
        for i in range(len(n_voxels)):  # 3
            l_conv = ConvModule(
                in_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        self.fpn_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        # Cache
        self.table = MappingTableV2()
        # self.data = {}

    def _init_points(self, pc_range, n_voxels):
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        for i, n_voxel in enumerate(n_voxels):
            x_voxels, y_voxels, z_voxels = n_voxel
            x = torch.linspace(x_min, x_max, x_voxels + 1)
            y = torch.linspace(y_min, y_max, y_voxels + 1)
            z = torch.linspace(z_min, z_max, z_voxels + 1)

            x_mid = (x[:-1] + x[1:]) / 2
            y_mid = (y[:-1] + y[1:]) / 2
            z_mid = (z[:-1] + z[1:]) / 2

            xx, yy, zz = torch.meshgrid(x_mid, y_mid, z_mid, indexing='ij')
            coords = torch.stack([xx, yy, zz], dim=0)
            self.register_buffer(f'points_{i}', coords)

    @staticmethod
    def _compute_projection(img_meta, stride):
        projection = []
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["ego2img"])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3] @ img_meta['bda_mat'].inverse())
        return torch.stack(projection)

    def forward(self, mlvl_feats, img_metas):
        batch_size = len(img_metas)
        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  # 3
            stride_i = math.ceil(img_metas[0]['img_shape'][0][-2] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):  # 4
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["ego2img"] = img_meta["ego2img"][seq_id * 6:(seq_id + 1) * 6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id * 6:(seq_id + 1) * 6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    # Ego to pixels. (6,3,4)
                    projection = self._compute_projection(img_meta, stride_i).to(feat_i.device)

                    # Sampling points.
                    points = getattr(self, f'points_{lvl}')

                    volume, valid = self.backproject_vanilla((img_meta["index"], lvl),  # tag for cache.
                                                             feat_i[:, :, :height, :width],
                                                             points,
                                                             projection)
                    volume = volume.sum(dim=0)
                    valid = valid.sum(dim=0)
                    volume = volume / valid
                    valid = valid > 0
                    volume[:, ~valid[0]] = 0.0

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])

            volume_list = torch.cat(volume_list, dim=1)  # (bs,seq*c,vx,vy,vz) (1,256,200,200,6)
            bs, c, dx, dy, dz = volume_list.shape  # 1,256,200,200,16
            # (bs,c,dx,dy,dz)->(bs,dx,dy,dz,c)->(bs,dx,dy,dz*c)->(bs,dz*c,dx,dy) 1,1536,200,200
            volume_list = volume_list.permute(0, 2, 3, 4, 1).reshape(bs, dx, dy, dz * c).permute(0, 3, 1,
                                                                                                 2).contiguous()
            mlvl_volumes.append(volume_list)  # list([bs,dz*n_times*c,vx,vy])

        # H & S & C attention
        if self.use_height_attention:
            mlvl_volumes = [
                attention(mlvl_volumes[i])
                for i, attention in enumerate(self.attentions)
            ]

        # build laterals
        laterals = [
            lateral_conv(mlvl_volumes[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        num_layers = len(laterals)
        for i in range(num_layers - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        out = self.fpn_conv(laterals[0])  # (1,64,200,200)

        return out  # (1, 64, 200, 200)

    def mapping_table(self, tag, points, projection):
        # idx, lvl = tag
        if self.training:
            # ego_to_cam
            # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
            points_2d_3 = torch.bmm(projection, points)  # lidar2img
        else:
            points_2d_3 = self.table.get(tag)
        # points_2d_3 = torch.bmm(projection, points)

        # with h5py.File('caches/mapping_table.h5', 'a') as f:
        #     f.create_dataset(tag, data=points_2d_3.cpu().numpy())

        # self.data[lvl] = points_2d_3
        # if lvl == 2:
        #     with open(f'./caches/{idx}.pkl', 'wb') as file:
        #         pickle.dump(self.data, file)
        #     self.data = {}

        return points_2d_3

    def backproject_vanilla(self, tag, features, points, projection):
        n_images, n_channels, height, width = features.shape
        n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
        # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        # [6, 3, 480000] -> [6, 4, 480000]
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)

        # 3d->2d
        points_2d_3 = self.mapping_table(tag, points, projection)

        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
        z = points_2d_3[:, 2]  # [6, 480000]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
        volume = torch.zeros(
            (n_images, n_channels, points.shape[-1]), device=features.device
        ).type_as(features)  # [6, 64, 480000]
        for i in range(n_images):
            volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
        # [6, 64, 480000] -> [6, 64, 200, 200, 12]
        volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
        # [6, 480000] -> [6, 1, 200, 200, 12]
        valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
        return volume, valid
