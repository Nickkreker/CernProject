import os
import numpy as np

from torch.utils.data import Dataset
import torch


class CernDataset(Dataset):
    def __init__(self, folder, start_moment=0, end_moment=1, max_dataset_size=None, min_file_size=9):
        self.root_dir = folder
        self.start = start_moment
        self.end = end_moment
        self.paths = []

        for idx in os.listdir(folder):
            for jobresult in os.listdir(f'{folder}/{idx}'):
                if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') and
                    os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') // 1048576 > min_file_size):
                    self.paths.append(f'{idx}/{jobresult}')

        if max_dataset_size is not None:
            self.paths = self.paths[:max_dataset_size]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample_name = self.paths[index]
        sample_path = os.path.join(self.root_dir, sample_name)

        img = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        with open(f'{sample_path}/printing_VISHNew/results/snapshot_Ed.dat') as f:
            for idx, line in enumerate(f):
                if idx > 262 * self.start and idx < 262 * (self.start + 1):
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    img = np.hstack((img, t))
                if idx > 262 * self.end and idx < 262 * (self.end + 1):
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    y = np.hstack((y, t))
        y_t = torch.from_numpy(y.reshape((1, 261, -1)))
        img_t = torch.from_numpy(img.reshape((1, 261, -1)))
        y_t = y_t[:, 3:-2, 3:-2]
        img_t = img_t[:, 3:-2, 3:-2]

        return img_t, y_t

class CernDatasetFullEvo(Dataset):
    def __init__(self, folder, evo_length=9, max_dataset_size=None):
        self.root_dir = folder
        self.evo_length = evo_length
        self.paths = []

        for idx in os.listdir(folder):
            for jobresult in os.listdir(f'{folder}/{idx}'):
                if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') and
                    os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') // 1048576 > evo_length + 1):
                    self.paths.append(f'{idx}/{jobresult}')

        if max_dataset_size is not None:
            self.paths = self.paths[:max_dataset_size]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample_name = self.paths[index]
        sample_path = os.path.join(self.root_dir, sample_name)

        img = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        with open(f'{sample_path}/printing_VISHNew/results/snapshot_Ed.dat') as f:
            for idx, line in enumerate(f):
                if idx > 262 and idx < 262 * 2:
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    img = np.hstack((img, t))
                if idx > 262 * 2 and idx < 262 * (self.evo_length + 2) and (idx % 262 != 0):
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    y = np.hstack((y, t))
        y_t = torch.from_numpy(y.reshape((self.evo_length, 261, 261)))
        img_t = torch.from_numpy(img.reshape((1, 261, 261)))
        y_t = y_t[:, 3:-2, 3:-2]
        img_t = img_t[:, 3:-2, 3:-2]

        return img_t, y_t
