import pickle
import threading
import time
from threading import Condition

TOTAL_CACHE_SIZE = 6019


class MappingTableV2:

    def __init__(self,
                 cache_path='/home/qingyuan/caches',
                 max_cache_size=256):
        # 最大缓存量
        self.max_cache_size = max_cache_size

        # 缓存最大存储 384
        self.caches = {}

        # 当前游标
        self.cur = 0

        # 加载完成等待唤醒继续加载
        self.condition = Condition()

        # 启动线程加载数据
        loader = threading.Thread(target=self._loading, args=(cache_path,))
        loader.daemon = True
        loader.start()

    def _loading(self, cache_path):
        while self.cur < TOTAL_CACHE_SIZE:
            # 加载数据到缓存
            for sample_idx in range(self.cur, min(self.cur + self.max_cache_size, TOTAL_CACHE_SIZE)):
                with open(f'{cache_path}/{sample_idx}.pkl', 'rb') as f:
                    sample = pickle.load(f)
                    self.caches[sample_idx] = sample
            # 更新当前游标
            self.cur = min(self.cur + self.max_cache_size, TOTAL_CACHE_SIZE)
            with self.condition:  # 等待通知继续加载数据到缓存
                self.condition.wait()

    def get(self, tag):
        sample_idx, lvl = tag

        # TODO 可能会死循环bug
        while sample_idx not in self.caches:  # 防止加载不及
            time.sleep(0.01)
            continue

        data = self.caches[sample_idx][lvl]

        if (sample_idx - 4) in self.caches:  # 移除过时帧，防止现存溢出
            del self.caches[sample_idx - 4]
            if len(self.caches) < (self.max_cache_size >> 2) and self.cur < TOTAL_CACHE_SIZE:  # < 64
                with self.condition:  # 开始加载数据
                    self.condition.notify()

        return data
