import numpy as np
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class MiniDataset(Dataset):
    def __init__(self, data, options, is_train=False):
        super(MiniDataset, self).__init__()
        self.data = np.array(data['x'])
        self.labels = np.array(data['y'])
        if self.labels.ndim > 1:
            self.labels = self.labels.squeeze(1)
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(), ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image, target = self.data[index], self.labels[index]
        image = image.reshape(32, 32, 3)
        image = self.transform(image)
        return image.float(), target
