import os.path as osp

import mmcv
from tqdm import tqdm

occ_train_path = 'data/occ3d-nus/occ_infos_temporal_train.pkl'
occ_val_path = 'data/occ3d-nus/occ_infos_temporal_val.pkl'

fastbev_train_path = 'data/occ3d-nus/nuscenes_infos_train_4d_interval3_max60.pkl'
fastbev_val_path = 'data/occ3d-nus/nuscenes_infos_val_4d_interval3_max60.pkl'


def _fill_trainval_infos(occ_path, fastocc_path):
    occ_infos, occ_metas = mmcv.load(occ_path)['infos'], mmcv.load(occ_path)['metadata']
    fastocc_infos = mmcv.load(fastocc_path)['infos']

    for idx in tqdm(range(len(occ_infos))):
        occ_info = occ_infos[idx]
        fastocc_info = fastocc_infos[idx]

        assert occ_info['token'] == fastocc_info['token'], \
            "{}, {}".format(occ_info['token'], fastocc_info['token'])

        fastocc_prevs = fastocc_info['prev']
        fastocc_nexts = fastocc_info['next']

        if fastocc_prevs is not None:
            for idx_p in range(len(fastocc_prevs)):  # prev
                fastocc_prevs[idx_p]['cams']['CAM_FRONT']['data_path'] = fastocc_prevs[idx_p]['cams']['CAM_FRONT'][
                    'data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_prevs[idx_p]['cams']['CAM_FRONT_RIGHT']['data_path'] = \
                    fastocc_prevs[idx_p]['cams']['CAM_FRONT_RIGHT']['data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_prevs[idx_p]['cams']['CAM_FRONT_LEFT']['data_path'] = \
                    fastocc_prevs[idx_p]['cams']['CAM_FRONT_LEFT']['data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_prevs[idx_p]['cams']['CAM_BACK']['data_path'] = fastocc_prevs[idx_p]['cams']['CAM_BACK'][
                    'data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_prevs[idx_p]['cams']['CAM_BACK_LEFT']['data_path'] = fastocc_prevs[idx_p]['cams']['CAM_BACK_LEFT'][
                    'data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_prevs[idx_p]['cams']['CAM_BACK_RIGHT']['data_path'] = \
                    fastocc_prevs[idx_p]['cams']['CAM_BACK_RIGHT']['data_path'].replace('/nuscenes/', '/occ3d-nus/')

        if fastocc_nexts is not None:
            for idx_n in range(len(fastocc_nexts)):  # next
                fastocc_nexts[idx_n]['cams']['CAM_FRONT']['data_path'] = fastocc_nexts[idx_n]['cams']['CAM_FRONT'][
                    'data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_nexts[idx_n]['cams']['CAM_FRONT_RIGHT']['data_path'] = \
                    fastocc_nexts[idx_n]['cams']['CAM_FRONT_RIGHT']['data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_nexts[idx_n]['cams']['CAM_FRONT_LEFT']['data_path'] = \
                    fastocc_nexts[idx_n]['cams']['CAM_FRONT_LEFT']['data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_nexts[idx_n]['cams']['CAM_BACK']['data_path'] = fastocc_nexts[idx_n]['cams']['CAM_BACK'][
                    'data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_nexts[idx_n]['cams']['CAM_BACK_LEFT']['data_path'] = fastocc_nexts[idx_n]['cams']['CAM_BACK_LEFT'][
                    'data_path'].replace('/nuscenes/', '/occ3d-nus/')
                fastocc_nexts[idx_n]['cams']['CAM_BACK_RIGHT']['data_path'] = \
                    fastocc_nexts[idx_n]['cams']['CAM_BACK_RIGHT']['data_path'].replace('/nuscenes/', '/occ3d-nus/')

        occ_info['prev'] = fastocc_prevs
        occ_info['next'] = fastocc_nexts
        occ_info['velo'] = fastocc_info['velo']

    return occ_infos, occ_metas


def patch_occ_pkl_with_prev_next():
    print('patch train...')
    occ_train_infos, metas = _fill_trainval_infos(occ_train_path, fastbev_train_path)
    data = dict(infos=occ_train_infos, metadata=metas)
    info_path = osp.join('data/occ3d-nus', '{}_infos_temporal_train.pkl'.format('fastocc'))
    mmcv.dump(data, info_path)

    print('patch val...')
    occ_val_infos, metas = _fill_trainval_infos(occ_val_path, fastbev_val_path)
    data = dict(infos=occ_val_infos, metadata=metas)
    info_path = osp.join('data/occ3d-nus', '{}_infos_temporal_val.pkl'.format('fastocc'))
    mmcv.dump(data, info_path)


if __name__ == '__main__':
    patch_occ_pkl_with_prev_next()

    # fastocc_train_pkl = mmcv.load('data/occ3d-nus/fastocc_infos_temporal_train.pkl')
    # len(fastocc_train_pkl)
    #
    # fastocc_val_pkl = mmcv.load('data/occ3d-nus/fastocc_infos_temporal_val.pkl')
    # len(fastocc_val_pkl)