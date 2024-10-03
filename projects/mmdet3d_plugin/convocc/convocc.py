# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize


@DETECTORS.register_module()
class ConvOcc(BaseDetector):
    def __init__(self,
                 # Modules
                 img_backbone,
                 img_neck,
                 neck_fuse,
                 img_view_transformer,
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 bbox_head,

                 use_ffm,
                 n_layers,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        # Modules
        self.backbone = build_backbone(img_backbone)
        self.neck = build_neck(img_neck)
        self.img_view_transformer = build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)
        self.bbox_head = build_head(bbox_head)

        # FFM
        for i in range(n_layers):
            self.add_module(
                f'neck_fuse_{i}',
                nn.Conv2d(neck_fuse['in_channels'][i],
                          neck_fuse['out_channels'][i],
                          3,
                          1,
                          1)
            )

        self.use_ffm = use_ffm
        self.n_layers = n_layers

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        # (1,256,200,200)->(1,128*i,100/i,100/i),i=1,2,4
        x = self.img_bev_encoder_backbone(x)
        # ...->(1,256,200,200)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_feat(self, img, img_metas=None):

        mlvl_feats = self.extract_img_feat(img)

        # (1,64,200,200)
        x = self.img_view_transformer(mlvl_feats, img_metas)

        # (1,256,200,200)
        x = self.bev_encoder(x)

        return x

    def extract_img_feat(self, img):
        # (4,24,3,900,1600)->(24,3,900,1600)
        img = img.view([-1] + list(img.shape)[2:])
        # (24,256*i,64/i,176/i) i=1,2,4,8
        x = self.backbone(img)
        # (24,64,64/i,176/i),i=1,2,4,8
        mlvl_feats = self.neck(x)

        mlvl_feats = self.ffm(mlvl_feats)

        return mlvl_feats

    def ffm(self, mlvl_feats):
        mlvl_feats_ = []
        if self.use_ffm:
            for msid in range(self.n_layers):
                # fpn output fusion
                fuse_feats = [mlvl_feats[msid]]
                for i in range(msid + 1, len(mlvl_feats)):
                    resized_feat = resize(
                        mlvl_feats[i],
                        size=mlvl_feats[msid].size()[2:],
                        mode="bilinear",
                        align_corners=False)
                    fuse_feats.append(resized_feat)

                fuse_feats = torch.cat(fuse_feats, dim=1)
                fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                mlvl_feats_.append(fuse_feats)  # (24,64,64/i,176/i),i=1,2,4
        else:
            mlvl_feats_ = mlvl_feats[:self.n_layers]
        return mlvl_feats_

    def forward_train(self,
                      img,  # (1,24,3,256,704)
                      img_metas,
                      mask_lidar=None,  # (1,200,200,16)
                      mask_camera=None,  # (1,200,200,16)
                      voxel_semantics=None,  # (1,200,200,16)
                      **kwargs):

        # (1,256,200,200) - bs,c,dx,dy
        feature_bev = self.extract_feat(img, img_metas)

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_occ = self.bbox_head.loss(voxel_semantics, mask_camera, x)
            losses.update(loss_occ)

        return losses

    def forward_test(self, img, img_metas, **kwargs):

        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas, **kwargs):

        # (1,256,200,200)
        feature_bev = self.extract_feat(img, img_metas)

        # (1,200,200,16,18)
        x = self.bbox_head(feature_bev)
        occ = self.bbox_head.get_occ(x)

        return occ

    def aug_test(self, imgs, img_metas, **kwargs):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):
            img_metas[0]['img_shape'] = img_shape_copy[24 * tta_id:24 * (tta_id + 1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24 * tta_id:24 * (tta_id + 1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24 * tta_id:24 * (tta_id + 1)], img_metas)
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass

    def forward_dummy(self,
                      img,  # (1,24,3,256,704)
                      img_metas,
                      **kwargs):

        # (1,256,200,200) - bs,c,dx,dy
        feature_bev = self.extract_feat(img, img_metas)

        x = self.bbox_head(feature_bev)

        return x
