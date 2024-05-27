_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_config = {
    'src_size': (900, 1600),
    'input_size': (928, 1600),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (928, 1600),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# 消融实验时会有某些参数未使用报错，设置为True忽略错误
# 不能与 with_cp = True 共同使用
# find_unused_parameters = True

# 是否使用 img feat encoder
use_img_feat_encoder = True

# 是否使用 multi-scale bev fusion
use_multi_bev = True

# 是否使用 height-attention
use_attention = True

# 是否开启时间融合
sequential = True
# 融合帧数
adj_ids = [1, 3, 5]  # 3帧

# batch_size
samples_per_gpu = 1

# 勿动
n_times = len(adj_ids) + 1 if sequential else 1
multi_scale_id = [0, 1, 2]  # 4x/8x/16x

model = dict(
    type='FastBEV',
    multi_scale_id=multi_scale_id,  # 4x
    use_img_feat_encoder=use_img_feat_encoder,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        with_cp=True,
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    neck_fuse=dict(in_channels=[256, 192, 128], out_channels=[64, 64 ,64]),
    img_view_transformer=dict(
        type='FastOccLSViewTransformer',
        in_channels=64 * n_times * 8,  # (c,n_times,dz)
        out_channels=64,
        n_voxels=[
            [200, 200, 8],  # 4x
            [150, 150, 8],  # 8x
            [100, 100, 8],  # 16x
        ],
        voxel_size=[
            [0.4, 0.4, 0.8],  # 4x
            [8 / 15, 8 / 15, 0.8],  # 8x
            [0.8, 0.8, 0.8],  # 16x
        ],
        back_project='mean',
        extrinsic_noise=0,
        multi_scale_3d_scaler='upsample',
        use_attention=use_attention,
        use_multi_bev=use_multi_bev,
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=64,
        num_channels=[64 * 2, 64 * 4, 64 * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=64 * 8 + 64 * 2,
        out_channels=256),
    bbox_head=dict(
        type='OccHead',
        bev_h=200,
        bev_w=200,
        pillar_h=16,
        num_classes=18,
        in_dims=256,
        out_dim=256,
        use_mask=True,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        use_class_weights=False,
    ),
)

dataset_type = 'InternalNuSceneOcc'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root = 'data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'fastocc_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        sequential=True,
        n_times=n_times,
        train_adj_ids=adj_ids,
        max_interval=10,
        min_interval=0,
        prev_only=True,
        test_adj='prev',
        test_adj_ids=adj_ids,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'fastocc_infos_temporal_val.pkl',
             pipeline=test_pipeline,
             classes=class_names,
             modality=input_modality,
             samples_per_gpu=1,
             sequential=True,
             n_times=n_times,
             train_adj_ids=adj_ids,
             max_interval=10,
             min_interval=0,
             test_adj='prev',
             test_adj_ids=adj_ids,
             ),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'fastocc_infos_temporal_val.pkl',
              pipeline=test_pipeline,
              classes=class_names,
              modality=input_modality,
              sequential=sequential,
              n_times=n_times,
              train_adj_ids=adj_ids,
              max_interval=10,
              min_interval=0,
              test_adj='prev',
              test_adj_ids=adj_ids,
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False
)

total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = './ckpts/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale='dynamic')

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=1,  # save only at epochs 2,4,6,...
        # resume='../work_dirs/efficient_occ_base_r50_e24_bda/epoch_20_ema.pth'
    ),
]
