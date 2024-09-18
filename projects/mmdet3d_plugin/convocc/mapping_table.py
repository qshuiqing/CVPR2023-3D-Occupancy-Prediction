# import pickle
# import threading
# import time
#
# total_cache_size = 6019
#
#
# class MappingTable:
#
#     def __init__(self,
#                  file_path='/home/qingyuan/caches',
#                  max_cache_size=256):
#         self.caches = {}
#         self.max_cache_size = max_cache_size
#         # 启动线程追加数据
#         threading.Thread(target=self._loader_worker, args=(file_path,)).start()
#
#     def _loader_worker(self, file_path):
#         sample_idx = 0
#         while sample_idx < self.max_cache_size:
#             with open(f'{file_path}/{sample_idx}.pkl', 'rb') as f:
#                 sample = pickle.load(f)
#                 self.caches[sample_idx] = sample
#                 sample_idx
#
#     def get(self, tag):
#         sample_idx, lvl = tag
#         while sample_idx not in self.caches:
#             time.sleep(0.01)
#             continue
#         data = self.caches[sample_idx][lvl]
#         if sample_idx - 4 in self.caches:
#             del self.caches[sample_idx - 4]
#         return data
#
# # if __name__ == '__main__':
# #
# #     lvl_0 = None
# #     lvl_1 = None
# #     lvl_2 = None
# #     for idx in tqdm(range(6019)):
# #         with open(f'/home/qingyuan/caches/{idx}.pkl', 'rb') as f:
# #             data = pickle.load(f)
# #             lvl_0 = data[0]
# #             lvl_1 = data[1]
# #             lvl_2 = data[2]
# #
# #     print(lvl_0.shape, lvl_1.shape, lvl_2.shape)
