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
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):

        super(OccHead, self).__init__()

        self.fp16_enabled = False
        self.num_classes = kwargs['num_classes']
        self.use_mask = use_mask

        self.loss_occ = build_loss(loss_occ)

        self.bev_h, self.bev_w = bev_h, bev_w

        if not use_3d:
            if use_conv:
                use_bias = norm_cfg is None
                self.decoder = nn.Sequential(
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg), )

            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims * 2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
                )
        else:
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

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, bev_embed):
        bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, self.bev_h, self.bev_w)
        if self.use_3d:
            outputs = self.decoder(bev_embed.view(bs, -1, self.pillar_h, self.bev_h, self.bev_w))
            outputs = outputs.permute(0, 4, 3, 2, 1)

        elif self.use_conv:

            outputs = self.decoder(bev_embed)
            outputs = outputs.view(bs, -1, self.pillar_h, bev_h, bev_w).permute(0, 3, 4, 2, 1)
        else:
            outputs = self.decoder(bev_embed.permute(0, 2, 3, 1))
            outputs = outputs.view(bs, bev_h, bev_w, self.pillar_h, self.out_dim)
        outputs = self.predicter(outputs)
        # print('outputs',type(outputs))
        return bev_embed, outputs

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             mask_camera,
             preds_dicts):

        loss_dict = dict()
        occ = preds_dicts['occ']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        losses = self.loss_single(voxel_semantics, mask_camera, occ)
        loss_dict['loss_occ'] = losses
        return loss_dict

    def loss_single(self, voxel_semantics, mask_camera, preds):
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out = preds_dicts['occ']
        occ_score = occ_out.softmax(-1)
        occ_score = occ_score.argmax(-1)

        return occ_score
