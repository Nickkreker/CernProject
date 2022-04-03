import os
import sys
import numpy as np

from torch.utils.data import Dataset
import torch


class CernDataset(Dataset):
    def __init__(self, folder, start_moment=0, end_moment=1, max_dataset_size=None, min_file_size=9, load_from_npy=False):
        self.root_dir = folder
        self.start = start_moment
        self.end = end_moment
        self.load_from_npy = load_from_npy
        self.paths = []

        for idx in os.listdir(folder):
            for jobresult in os.listdir(f'{folder}/{idx}'):
                if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') and
                    os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') // 1048576 > min_file_size):
                    self.paths.append(f'{idx}/{jobresult}')

        if self.load_from_npy:
            paths_t = []
            for path in self.paths:
                path_t = f'{folder}/{path}/printing_VISHNew/results'
                if (os.path.exists(f'{path_t}/y.npy') and
                    os.path.getsize(f'{path_t}/y.npy') == 2452484):
                    paths_t.append(path)
            self.paths = paths_t


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
            img_t = np.reshape(np.load(f'{sample_path}/printing_VISHNew/results/img.npy'), (1, 261, 261))
            y_t = np.reshape(np.load(f'{sample_path}/printing_VISHNew/results/y.npy'), (-1, 261, 261))
            if self.start == 0:
                img_t = img_t
                y_t = y_t[self.end - 1]
            else:
                img_t = y_t[self.start - 1]
                y_t = y_t[self.end - 1]
            y_t = torch.from_numpy(y_t.reshape((1, 261, -1)))
            img_t = torch.from_numpy(img_t.reshape((1, 261, -1)))    
        else:
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
    def __init__(self, folder, evo_length=9, max_dataset_size=None, load_from_npy=False, load_velocities=False, modified_velocities=True):
        self.root_dir = folder
        self.evo_length = evo_length
        self.load_from_npy = load_from_npy
        self.load_velocities = load_velocities
        self.modified_velocities = modified_velocities
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

        if load_velocities:
            paths_t = []
            for path in self.paths:
                path_t = f'{folder}/{path}/printing_VISHNew/results'
                if (os.path.exists(f'{path_t}/vx.npy') and
                    os.path.exists(f'{path_t}/vy.npy') and
                    os.path.getsize(f'{path_t}/vx.npy') == 2452484 and 
                    os.path.getsize(f'{path_t}/vy.npy') == 2452484):
                    paths_t.append(path)
            self.paths = paths_t

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

        if self.load_velocities:
            vxs = np.load(f'{sample_path}/printing_VISHNew/results/vx.npy')
            vys = np.load(f'{sample_path}/printing_VISHNew/results/vy.npy')
            y = np.concatenate((y, vxs))
            y = np.concatenate((y, vys))

        y_t = torch.from_numpy(y.reshape((-1, 261, 261)))
        img_t = torch.from_numpy(img.reshape((1, 261, 261)))

        if self.modified_velocities:
            y_t[9:18] *= y_t[:9]
            y_t[18:27] *= y_t[:9]
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


class CernDatasetOneStepVelocities(Dataset):
    def __init__(self, folder, start_moment=0, end_moment=1, flatten=False, predict_type=0, modified_velocities=False, max_dataset_size=None,
                 use_init_velocities=False):
        self.root_dir = folder
        self.flatten = flatten
        self.start_moment = start_moment
        self.end_moment = end_moment
        self.predict_type = predict_type
        self.modified_velocities = modified_velocities
        self.use_init_velocities = use_init_velocities
        self.paths = []

        for idx in os.listdir(folder):
            for jobresult in os.listdir(f'{folder}/{idx}'):
                if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vx.npy') and
                        os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vy.npy') and
                        os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/y.npy') and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vx.npy') == 2452484 and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vy.npy') == 2452484 and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/y.npy') == 2452484):
                        self.paths.append(f'{idx}/{jobresult}')

        if max_dataset_size is not None:
            self.paths = self.paths[:max_dataset_size]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample_name = self.paths[index]
        sample_path = os.path.join(self.root_dir, sample_name)

        ed_init = np.array([], dtype=np.float32)
        ed_final = np.array([], dtype=np.float32)
        vx_init = np.array([], dtype=np.float32)
        vx_final = np.array([], dtype=np.float32)
        vy_init = np.array([], dtype=np.float32)
        vy_final = np.array([], dtype=np.float32)

        ed_full = np.load(f'{sample_path}/printing_VISHNew/results/y.npy')
        ed_full = np.reshape(ed_full, (-1, 261, 261))
        ed_final = ed_full[self.end_moment-1]

        vx_full = np.load(f'{sample_path}/printing_VISHNew/results/vx.npy')
        vy_full = np.load(f'{sample_path}/printing_VISHNew/results/vy.npy')
        vx_full = np.reshape(vx_full, (-1, 261, 261))
        vy_full = np.reshape(vy_full, (-1, 261, 261))

        if self.start_moment == 0:
            ed_init = np.load(f'{sample_path}/printing_VISHNew/results/img.npy')
            ed_init = np.reshape(ed_init, (261, 261))
            vx_init = np.zeros_like(ed_init)
            vy_init = np.zeros_like(ed_init)
        else:
            ed_init = ed_full[self.start_moment-1]
            vx_init = vx_full[self.start_moment-1]
            vy_init = vy_full[self.start_moment-1]

        vx_final = vx_full[self.end_moment-1]
        vy_final = vy_full[self.end_moment-1]

        # Normalization of velocities (delete after test)
        # vx_init -= np.min(vx_init)
        # vy_init -= np.min(vy_init)

        # V_x = V_x * Ed, V_y = V_y * Ed
        if self.modified_velocities:
            vx_init = vx_init * ed_init
            vy_init = vy_init * ed_init
            vx_final = vx_final * ed_final
            vy_final = vy_final * ed_final

        x_t = torch.from_numpy(np.array((ed_init, vx_init, vy_init)))
        y_t = torch.from_numpy(np.array((ed_final, vx_final, vy_final)))

        x_t = x_t[:, 3:-2, 3:-2]
        y_t = y_t[:, 3:-2, 3:-2]

        if self.predict_type == 0:
            y_t = y_t[0]
        elif self.predict_type == 1:
            y_t = y_t[1]
        elif self.predict_type == 2:
            y_t = y_t[2]

        if not self.use_init_velocities:
            x_t = torch.unsqueeze(x_t[0], 0)

        if self.flatten:
            x_t = torch.unsqueeze(torch.reshape(x_t, (-1, 256)), 0)
            y_t = torch.unsqueeze(torch.reshape(y_t, (-1, 256)), 0)

        return x_t, y_t

    


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


class CernDatasetVelocities(Dataset):
    def __init__(self, folder, max_dataset_size=None, min_file_size=9, flatten=False, paths_from_npys=False):
        self.root_dir = folder
        self.paths = []
        self.num_time_stamps = min_file_size - 1
        self.flatten = flatten

        if paths_from_npys:
            for idx in os.listdir(folder):
                for jobresult in os.listdir(f'{folder}/{idx}'):
                    if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vx.npy') and
                        os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vy.npy') and
                        os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/y.npy') and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vx.npy') == 2452484 and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/vy.npy') == 2452484 and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/y.npy') == 2452484):
                        for time_stamp in range(self.num_time_stamps + 1):
                            self.paths.append(f'{idx}/{jobresult}{time_stamp}')
        else:
            for idx in os.listdir(folder):
                for jobresult in os.listdir(f'{folder}/{idx}'):
                    if (os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') and
                        os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Vy.dat') and
                        os.path.exists(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Vx.dat') and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Ed.dat') // 1048576 > min_file_size + 1 and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Vx.dat') // 1048576 > min_file_size + 1 and
                        os.path.getsize(f'{folder}/{idx}/{jobresult}/printing_VISHNew/results/snapshot_Vy.dat') // 1048576 > min_file_size + 1):
                        for time_stamp in range(self.num_time_stamps + 1):
                            self.paths.append(f'{idx}/{jobresult}{time_stamp}')

        if max_dataset_size is not None:
            self.paths = self.paths[:max_dataset_size]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample_name = self.paths[index][:-1]
        time_stamp = int(self.paths[index][-1])
        sample_path = os.path.join(self.root_dir, sample_name)

        ed_init = np.array([], dtype=np.float32)
        ed_final = np.array([], dtype=np.float32)
        vx_init = np.array([], dtype=np.float32)
        vx_final = np.array([], dtype=np.float32)
        vy_init = np.array([], dtype=np.float32)
        vy_final = np.array([], dtype=np.float32)

        ed_full = np.load(f'{sample_path}/printing_VISHNew/results/y.npy')
        ed_full = np.reshape(ed_full, (-1, 261, 261))
        ed_final = ed_full[time_stamp]

        vx_full = np.load(f'{sample_path}/printing_VISHNew/results/vx.npy')
        vy_full = np.load(f'{sample_path}/printing_VISHNew/results/vy.npy')
        vx_full = np.reshape(vx_full, (-1, 261, 261))
        vy_full = np.reshape(vy_full, (-1, 261, 261))

        if time_stamp == 0:
            ed_init = np.load(f'{sample_path}/printing_VISHNew/results/img.npy')
            ed_init = np.reshape(ed_init, (261, 261))
            vx_init = np.zeros_like(ed_init)
            vy_init = np.zeros_like(ed_init)
        else:
            ed_init = ed_full[time_stamp - 1]
            vx_init = vx_full[time_stamp - 1]
            vy_init = vy_full[time_stamp - 1]

        vx_final = vx_full[time_stamp]
        vy_final = vy_full[time_stamp]

        x_t = torch.from_numpy(np.array((ed_init, vx_init, vy_init)))
        y_t = torch.from_numpy(np.array((ed_final, vx_final, vy_final)))

        x_t = x_t[:, 3:-2, 3:-2]
        y_t = y_t[:, 3:-2, 3:-2]

        if self.flatten:
            x_t = torch.unsqueeze(torch.reshape(x_t, (-1, 256)), 0)
            y_t = torch.unsqueeze(torch.reshape(y_t, (-1, 256)), 0)

        return x_t, y_t

    def generate_npys(self):
        """Generates vx.npy and vy.npy for each training sample for faster load."""
        num_paths = len(self.paths) // self.num_time_stamps
        for i, path in enumerate(self.paths):
            if i % (self.num_time_stamps + 1) != 0:
                continue
            path = path[:-1]
            sample_path = os.path.join(self.root_dir, path)
            print(path)

            vx = np.array([], dtype=np.float32)
            vy = np.array([], dtype=np.float32)

            with open(f'{sample_path}/printing_VISHNew/results/snapshot_Vx.dat') as f:
                for idx, line in enumerate(f):
                    if idx > 262 * 2 and idx < 262 * (self.num_time_stamps + 3) and (idx % 262 != 0):
                        t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                        vx = np.hstack((vx, t))

            # Checking whether the file is broken
            try:
                np.reshape(vx, (-1, 261, 261))
            except:
                print(f'{sample_path}/printing_VISHNew/results/snapshot_Vx.dat')

            with open(f'{sample_path}/printing_VISHNew/results/snapshot_Vy.dat') as f:
                for idx, line in enumerate(f):
                    if idx > 262 * 2 and idx < 262 * (self.num_time_stamps + 3) and (idx % 262 != 0):
                        t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
                        vy = np.hstack((vy, t))

            try:
                np.reshape(vy, (-1, 261, 261))
            except:
                print(f'{sample_path}/printing_VISHNew/results/snapshot_Vy.dat')

            np.save(f'{sample_path}/printing_VISHNew/results/vx.npy',  vx)
            np.save(f'{sample_path}/printing_VISHNew/results/vy.npy',  vy)

            print(f'{i // self.num_time_stamps + 1}/{num_paths} generated')
