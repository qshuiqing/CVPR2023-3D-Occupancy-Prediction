import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models import NECKS


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
class FastOccLSViewTransformer(BaseModule):
    def __init__(self,
                 n_voxels,
                 voxel_size,
                 in_channels,
                 out_channels,
                 back_project,
                 extrinsic_noise,
                 multi_scale_3d_scaler,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 ):
        super(FastOccLSViewTransformer, self).__init__()

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.back_project = back_project
        self.multi_scale_3d_scaler = multi_scale_3d_scaler
        self.upsample_cfg = upsample_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        self.attentions = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        for i in range(len(n_voxels)):  # 3

            self.attentions.append(BasicBlock(in_channels))

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

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["ego2img"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats(list[torch.Tensor]): of shape (bs*nc*n_times, c, h_feat, w_feat).
            img_metas(list[dict]): Meta information of samples.

        Returns:
            list[torch.Tensor]: of shape (bs, c, dx, dy).
        """
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

                    projection = self._compute_projection(  # (6,3,4)
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    # v3/v4 bev ms
                    n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["origin"]),
                    ).to(feat_i.device)

                    if self.back_project == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)
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

        # C & S & C attention
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


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
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


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume
