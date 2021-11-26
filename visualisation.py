import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def visualize_dataset(dataset, title=None, count=10, visualize_y=False):
    '''
    Visualizes initial states from dataset
    '''
    display_indices = np.random.choice(np.arange(len(dataset)), count, replace=False)
    plt.figure(figsize=(count * 3, (visualize_y + 1) * 3))
    if title:
        plt.suptitle("%s %s/%s" % (title, len(display_indices), len(indices)))
    for i, index in enumerate(display_indices):
        x, y = dataset[index]
        if visualize_y:
            plt.subplot(2, count, i + 1)
            plt.imshow(x.squeeze())
            plt.subplot(2, count, i + count + 1)
            plt.imshow(y.squeeze())
        else:
            plt.subplot(1, count, i + 1)
            plt.imshow(x.squeeze())

def visualise_single_step_prediction(model, dataset, device=torch.device('cuda:0'), count=5):
    '''
    Plots actual and predicted states
    '''
    model.eval()

    indices = list(range(count))

    sampler = SubsetSampler(indices)
    loader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler)
    fig = plt.figure(figsize=(16,12), dpi=250)
    fig.tight_layout()
    for counter, (x, y) in enumerate(loader):
        if counter > count:
            return
        x_gpu = x.to(device)
        prediction_gpu = model(x_gpu)
        prediction = prediction_gpu.cpu().detach()
        fig.add_subplot(3, count, counter + 1)
        plt.title("initial")
        plt.imshow(torch.squeeze(x))
        fig.add_subplot(3, count, counter + count + 1)
        plt.title("predicted")
        plt.imshow(torch.squeeze(prediction))
        fig.add_subplot(3, count, counter + 2 * count + 1)
        plt.title("actual")
        plt.imshow(torch.squeeze(y))
    # plt.savefig('../plots/unet_adam_5ep.png')

def visualise_full_evo(model, dataset, device=torch.device('cuda:0'), count=5):
    model.eval()

    indices = list(range(count))

    sampler = SubsetSampler(indices)
    loader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler)

    fig = plt.figure(figsize=(27,9), dpi=250)

    for counter, (x, y) in enumerate(loader):
        if counter > 0:
            break

        t = x.to(device)
        for k in range(9):
            prediction_gpu = model(t)
            prediction = prediction_gpu.cpu().detach()
            fig.add_subplot(3, 9, k + 1)
            plt.title("initial")
            plt.imshow(torch.squeeze(x))
            fig.add_subplot(3, 9, k + 9 + 1)
            plt.title("predicted")
            plt.imshow(torch.squeeze(prediction))
            fig.add_subplot(3, 9, k + 18 + 1)
            plt.title("actual")
            plt.imshow(torch.squeeze(y[:,k,:, :]))
            t = prediction_gpu

            #plt.savefig('../plots/full_evo_prediction_bad.png')


def plot_losses(train_loss_history, val_loss_history, path_to_losses=None, output_filepath=None):
    if path_to_losses is not None:
        train_loss_history = []
        val_loss_history = []
        with open(path_to_losses, "r") as file:
            for line if file:
                train_loss_history.append(float(line.split(' ')[2][:-1]))
                val_loss_history.append(float(line.split(' ')[5]))

    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('model loss')
    plt.legend(loc='upper right')
    plt.show()

    if output_filepath is not None:
        plt.savefig(output_filepath)
