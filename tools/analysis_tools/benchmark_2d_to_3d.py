# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


# from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--samples', default=1000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg.data.test)
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #    model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 500
    pure_inf_time = 0

    model = model.module
    prev_bev = model.prev_frame_info['prev_bev']

    occ_head = model.pts_bbox_head
    dtype = torch.float32
    bev_queries = occ_head.bev_embedding.weight.to(dtype)
    bev_mask = torch.zeros((1, 200, 200),
                           device=bev_queries.device).to(dtype)
    bev_pos = occ_head.positional_encoding(bev_mask).to(dtype)

    transformer_occ = model.pts_bbox_head.transformer

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        img = data['img'][0].data[0].cuda()
        img_metas = data['img_metas'][0].data[0]

        with torch.no_grad():
            mlvl_feats = model.extract_img_feat(img, img_metas)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            bev_feats = transformer_occ.get_bev_features(mlvl_feats,
                                                         bev_queries,
                                                         200,
                                                         200,
                                                         grid_length=(0.4, 0.4),
                                                         bev_pos=bev_pos,
                                                         prev_bev=prev_bev,
                                                         img_metas=img_metas)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            # if (i + 1) % args.log_interval == 0:
            #     fps = (i + 1 - num_warmup) / pure_inf_time
            #     print(f'Done image [{i + 1:<3}/ {args.samples}], '
            #           f'fps: {fps:.2f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall \nfps: {fps:.2f} img / s '
                  f'\ninference time: {1000 / fps:.2f} ms')
            break


if __name__ == '__main__':
    main()