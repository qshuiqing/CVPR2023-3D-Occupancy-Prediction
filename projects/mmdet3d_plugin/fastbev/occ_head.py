# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmcv.runner import auto_fp16
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from torch import nn

from projects.mmdet3d_plugin.fastbev.loss.lovasz_softmax import lovasz_softmax


@HEADS.register_module()
class OccHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 loss_occ=None,
                 use_mask=False,
                 num_classes=18,
                 pillar_h=16,
                 use_3d=False,
                 use_conv=False,
                 embed_dims=256,
                 out_dim=32,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):

        super(OccHead, self).__init__()

        self.fp16_enabled = False
        self.num_classes = num_classes
        self.use_mask = use_mask

        self.loss_occ = build_loss(loss_occ)

        self.bev_h, self.bev_w = bev_h, bev_w

        self.embed_dims = embed_dims
        self.out_dim = out_dim

        self.pillar_h = pillar_h
        self.use_3d = use_3d
        self.use_conv = use_conv

        use_bias_3d = norm_cfg_3d is None
        self.middle_dims = self.embed_dims // pillar_h
        self.decoder = nn.Sequential(
            ConvModule(
                self.middle_dims,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias_3d,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg_3d,
                act_cfg=act_cfg),
            ConvModule(
                self.out_dim,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias_3d,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg_3d,
                act_cfg=act_cfg),
        )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, num_classes),
        )

    @auto_fp16(apply_to='bev_embed')
    def forward(self, bev_embed):
        """

        Args:
            bev_embed: (1,256,100,100) - bs,c,dx,dy

        Returns:

        """
        bs, c, dx, dy = bev_embed.shape

        # (1,16,16,200,200)->(1,32,16,200,200)
        outputs = self.decoder(bev_embed.view(bs, -1, self.pillar_h, dx, dy))
        outputs = outputs.permute(0, 3, 4, 2, 1)  # (1,200,200,16,32) - bs,dx,dy,dz,c

        # (1,200,200,16,18)
        outputs = self.predicter(outputs)

        return outputs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             mask_camera,
             preds_dicts):

        loss_dict = dict()
        occ = preds_dicts
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ, loss_lovasz = self.loss_single(voxel_semantics, mask_camera, occ)
        loss_dict['loss_occ'], loss_dict['loss_lovasz'] = loss_occ, loss_lovasz
        return loss_dict

    def loss_single(self, voxel_semantics, mask_camera, preds):
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)

            voxel_semantics = voxel_semantics[mask_camera.nonzero().squeeze()]
            preds = preds[mask_camera.nonzero().squeeze()]
            loss_lovasz = lovasz_softmax(preds.softmax(-1), voxel_semantics)

        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_lovasz = lovasz_softmax(preds.softmax(-1), voxel_semantics)

        return loss_occ, loss_lovasz

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds):
        """Generate occupancy from occ head predictions.
        Args:
            preds_dicts : occ results.
        Returns:
            list[dict]: labels.
        """
        occ_out = preds
        occ_score = occ_out.softmax(-1)
        occ_score = occ_score.argmax(-1)

        return occ_score
