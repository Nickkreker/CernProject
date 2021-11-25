import os
import sys
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
                if idx > 262 * (self.end + 1) and idx < 262 * (self.end + 2):
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    y = np.hstack((y, t))
        y_t = torch.from_numpy(y.reshape((1, 261, -1)))
        img_t = torch.from_numpy(img.reshape((1, 261, -1)))
        y_t = y_t[:, 3:-2, 3:-2]
        img_t = img_t[:, 3:-2, 3:-2]

        return img_t, y_t

class CernDatasetFullEvo(Dataset):
    def __init__(self, folder, evo_length=9, max_dataset_size=None, load_from_npy=False):
        self.root_dir = folder
        self.evo_length = evo_length
        self.load_from_npy = load_from_npy
        self.paths = []

        broken_files = {
            'D:/CernDataset/447/jobresult_5/printing_VISHNew/results/snapshot_Ed.dat',
            'D:/CernDataset/802/jobresult_8/printing_VISHNew/results/snapshot_Ed.dat',
            'D:/CernDataset/985/jobresult_1/printing_VISHNew/results/snapshot_Ed.dat'
        }

        for idx in os.listdir(folder):
            for jobresult in os.listdir(f'{folder}/{idx}'):
                path = f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat'
                if (os.path.exists(path) and os.path.getsize(path) // 1048576 > evo_length + 1):
                    if path not in broken_files:
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

        if self.load_from_npy:
            y = np.load(f'{sample_path}/printing_VISHNew/results/y.npy')
            img = np.load(f'{sample_path}/printing_VISHNew/results/img.npy')
        else:
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

    def generate_npys(self):
        """Generates img.npy and y.npy for each training sample for faster load."""
        num_paths = len(self.paths)
        for i, path in enumerate(self.paths):
            sample_path = os.path.join(self.root_dir, path)

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

            np.save(f'{sample_path}/printing_VISHNew/results/y.npy',  y)
            np.save(f'{sample_path}/printing_VISHNew/results/img.npy',  img)

            print(f'{i+1}/{num_paths} generated')


class CernDatasetMassive(Dataset):
    def __init__(self, folder, max_dataset_size=None, min_file_size=9):
        self.root_dir = folder
        self.paths = []
        self.num_time_stamps = min_file_size - 1

        for idx in os.listdir(folder):
            for jobresult in os.listdir(f'{folder}/{idx}'):
                if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') and
                    os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') // 1048576 > min_file_size):
                    for time_stamp in range(self.num_time_stamps):
                        self.paths.append(f'{idx}/{jobresult}{time_stamp}')

        if max_dataset_size is not None:
            self.paths = self.paths[:max_dataset_size]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample_name = self.paths[index][:-1]
        time_stamp = int(self.paths[index][-1])
        sample_path = os.path.join(self.root_dir, sample_name)

        img = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        with open(f'{sample_path}/printing_VISHNew/results/snapshot_Ed.dat') as f:
            for idx, line in enumerate(f):
                if idx > 262 * (time_stamp + 1) and idx < 262 * (time_stamp + 2):
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    img = np.hstack((img, t))
                if idx > 262 * (time_stamp + 2) and idx < 262 * (time_stamp + 3):
                    t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                    y = np.hstack((y, t))
        y_t = torch.from_numpy(y.reshape((1, 261, -1)))
        img_t = torch.from_numpy(img.reshape((1, 261, -1)))
        y_t = y_t[:, 3:-2, 3:-2]
        img_t = img_t[:, 3:-2, 3:-2]

        return img_t, y_t
