import os
import random as prandom
import string

import cv2
import numpy as np
import torch
from PIL import Image
from mmdet.datasets.builder import PIPELINES

from projects.mmdet3d_plugin.core.visualizer.image_vis import draw_lidar_bbox3d_on_img


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
    ):
        self.data_root = data_root

    def __call__(self, results):
        # print(results.keys())
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root, occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']
        else:
            semantics = np.zeros((200, 200, 16), dtype=np.uint8)
            mask_lidar = np.zeros((200, 200, 16), dtype=np.uint8)
            mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        semantics = torch.from_numpy(semantics)
        mask_lidar = torch.from_numpy(mask_lidar)
        mask_camera = torch.from_numpy(mask_camera)

        if results.get('flip_dx', False):  # semantics[::-1,...]
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):  # semantics[:,::-1,...]
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])


        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@PIPELINES.register_module()
class RandomAugImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self,
                 data_config=None,
                 is_train=True,
                 is_debug=False,
                 is_exit=False,
                 tmp='./figs/augs'):
        self.data_config = data_config
        self.is_train = is_train
        self.is_debug = is_debug
        self.is_exit = is_exit
        self.tmp = tmp

    def random_id(self, N=8, seed=None):
        if seed is not None:
            prandom.seed(seed)
        return ''.join(prandom.choice(string.ascii_uppercase + string.digits) for _ in range(N))

    def __call__(self, results, fix=''):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        imgs = []
        for cam_id, img in enumerate(results['img']):
            pil_img = Image.fromarray(img, mode='RGB')
            resize, resize_dims, crop, flip, rotate, pad = self.sample_augmentation(
                H=pil_img.height,
                W=pil_img.width,
            )
            post_pil_img, post_rot, post_tran = self.img_transform(
                pil_img, torch.eye(2), torch.zeros(2),
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
                pad=pad
            )
            imgs.append(np.asarray(post_pil_img))
            results['lidar2img'][cam_id], results['ego2img'][cam_id] = \
                self.rts2proj(results['lidar2img'][cam_id], results['ego2img'][cam_id], post_rot, post_tran)
        results['img'] = imgs
        results['img_shape'] = [img.shape for img in results['img']]
        # tmp_usages for augmentated images
        if self.is_debug:
            lidar2imgs = results['lidar2img']
            imgs = results['img']
            bboxes = results['gt_bboxes_3d']
            bid = self.random_id(8, results['sample_idx']) + fix
            for ii in range(len(results['img'])):
                cam_id = ii % 6
                cam_type = {True: 'curr', False: 'adj'}[ii // 6 == 0]
                # if cam_type != 'curr':
                #     continue
                try:
                    new_img = draw_lidar_bbox3d_on_img(bboxes, imgs[ii], lidar2imgs[ii], dict())
                    img_filename = f'{self.tmp}/{results["index"]}/{cam_id}_{cam_type}_{ii // 6}_' + \
                                   results['img_filename'][ii].split('/')[-1]
                    if not os.path.exists(os.path.join(self.tmp, str(results["index"]))):
                        os.makedirs(os.path.join(self.tmp, str(results["index"])))
                    cv2.imwrite(img_filename, new_img)
                except Exception:
                    new_img = imgs[ii]
                    img_filename = f'{self.tmp}/{results["index"]}/{cam_id}_{cam_type}_{ii // 6}_err_' + \
                                   results['img_filename'][ii].split('/')[-1]
                    if not os.path.exists(os.path.join(self.tmp, str(results["index"]))):
                        os.makedirs(os.path.join(self.tmp, str(results["index"])))
                    cv2.imwrite(img_filename, new_img)
        if self.is_exit:
            exit()
        return results

    def sample_augmentation(self, H, W):
        if self.is_train:
            fH, fW = self.data_config['input_size']  # (640, 1600),
            resize = float(fW) / float(W)  # 1600 / 1600
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))  # 900 1600

            newW, newH = resize_dims  # 1600 900
            crop_h_start = (newH - fH) // 2  # (900 - 640)  // 2 = 130
            crop_w_start = (newW - fW) // 2  # 1600 1600
            crop_h_start += int(np.random.uniform(*self.data_config['crop']) * fH)
            crop_w_start += int(np.random.uniform(*self.data_config['crop']) * fW)

            # (0, 130, 1600, 130+640)
            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            fH, fW = self.data_config['test_input_size']
            resize = float(fW) / float(W)
            resize += self.data_config.get('test_resize', 0.0)
            resize_dims = (int(W * resize), int(H * resize))

            newW, newH = resize_dims
            crop_h_start = (newH - fH) // 2
            crop_w_start = (newW - fW) // 2
            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

            flip = self.data_config['test_flip']
            rotate = self.data_config['test_rotate']

        pad_data = self.data_config['pad']
        pad_divisor = self.data_config['pad_divisor']
        pad_color = self.data_config['pad_color']
        pad = (pad_data, pad_color)

        return resize, resize_dims, crop, flip, rotate, pad

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate, pad):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate, pad)

        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        top, right, bottom, left = pad[0]
        post_tran[0] = post_tran[0] + left  # left
        post_tran[1] = post_tran[1] + top  # top

        ret_post_rot, ret_post_tran = np.eye(3), np.zeros(3)
        ret_post_rot[:2, :2] = post_rot
        ret_post_tran[:2] = post_tran

        return img, ret_post_rot, ret_post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate, pad):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        top, right, bottom, left = pad[0]
        pad_color = pad[1]
        img = self.img_pad(img, top, right, bottom, left, pad_color)
        return img

    def img_pad(self, pil_img, top, right, bottom, left, color):
        if top == right == bottom == left == 0:
            return pil_img
        assert top == bottom, (top, right, bottom, left)
        assert left == right, (top, right, bottom, left)
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def rts2proj(self, lidar2img, ego2img, post_rot=None, post_tran=None):
        viewpad = np.eye(4)
        if post_rot is not None:
            assert post_tran is not None, [post_rot, post_tran]
            viewpad[:3, :3] = post_rot
            viewpad[:3, 2] += post_tran

        lidar2img_rt = np.array((viewpad @ lidar2img), dtype=np.float32)
        ego2img_rt = np.array((viewpad @ ego2img), dtype=np.float32)

        return lidar2img_rt.astype(np.float32), ego2img_rt.astype(np.float32)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['origin'] = self.origin.copy()
        return results

@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self, tta_config=None):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            if tta_config is not None:
                flip_dx = tta_config['flip_dx']
                flip_dy = tta_config['flip_dy']
            else:
                flip_dx = False
                flip_dy = False

        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        # if gt_boxes.shape[0] > 0:
        #     gt_boxes[:, :3] = (
        #             rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        #     gt_boxes[:, 3:6] *= scale_ratio
        #     gt_boxes[:, 6] += rotate_angle
        #     if flip_dx:
        #         gt_boxes[:,
        #         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
        #                                                  6]
        #     if flip_dy:
        #         gt_boxes[:, 6] = -gt_boxes[:, 6]
        #     gt_boxes[:, 7:] = (
        #             rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        # gt_boxes, gt_labels = results['ann_info']['gt_bboxes_3d'], results['ann_info']['gt_labels_3d']
        # gt_boxes, gt_labels = gt_boxes.tensor, torch.tensor(gt_labels)
        tta_confg = results.get('tta_config', None)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(tta_confg)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(None, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        # if len(gt_boxes) == 0:
        #     gt_boxes = torch.zeros(0, 9)
        # results['gt_bboxes_3d'] = \
        #     LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
        #                          origin=(0.5, 0.5, 0.5))
        # results['gt_labels_3d'] = gt_labels

        results['bda_mat'] = bda_mat

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        return results