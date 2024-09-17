import threading
import time

import h5py
import torch

total_cache_size = 6019


class MappingTable:

    def __init__(self,
                 file_path='./caches/mapping_tables.h5',
                 max_cache_size=256):
        self.lvl_0 = {}
        self.access_0 = set()
        self.lvl_1 = {}
        self.access_1 = set()
        self.lvl_2 = {}
        self.access_2 = set()

        # 游标，下条记录下标
        self.cur = 0

        # 加载初始数据
        self._init_load(file_path, max_cache_size)
        # 启动线程追加数据
        threading.Thread(target=self._loader_worker, args=(file_path, max_cache_size)).start()

    def _loader_worker(self, file_path, max_cache_size):
        while self.cur < total_cache_size:
            self._init_load(file_path, max_cache_size)

    def _init_load(self, file_path, max_cache_size):
        with h5py.File(file_path, 'r') as h5:
            pre_cur = self.cur
            self.cur = min(pre_cur + max_cache_size, total_cache_size)
            self._extend(h5, pre_cur, self.cur)

    def _extend(self, h5, start, end):
        self.lvl_0.update({i: h5['0'][i] for i in range(start, end)})
        self.lvl_1.update({i: h5['1'][i] for i in range(start, end)})
        self.lvl_2.update({i: h5['2'][i] for i in range(start, end)})

    def get(self, tag):
        idx, idx_lvl = tag
        lvl = getattr(self, f'lvl_{idx_lvl}')
        while idx not in lvl:
            time.sleep(0.01)  # 让其他线程有时间加载数据
            continue
        data = torch.Tensor(lvl[idx])

        access = getattr(self, f'access_{idx_lvl}')
        access.add(idx)

        if len(access) > 4:
            del lvl[access.pop()]

        return data


# if __name__ == '__main__':
#     cache_file = './outer/mapping_tables.h5'
#
#     mt = MappingTable(cache_file)
#
#     tags = [(i, j) for i in range(1024) for j in range(3)]
#     for i in range(total_cache_size * 3):
#         mt.get(tags[i])
#
#     print(mt.lvl_0.keys())
#     print(mt.lvl_1.keys())
#     print(mt.lvl_2.keys())
