import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import random
import numpy as np
from torchvision.datasets import MNIST
from PIL import Image
import torchvision
import IPython
from torchvision.utils import save_image
import math
import io
import torchvision.utils as vutils
import matplotlib.pyplot as plt;

plt.rcParams['figure.dpi'] = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dict_curr_colors = {0: (156, 102, 31), 1: (0, 0, 255), 2: (127, 255, 0), 3: (255, 20, 147), 4: (255, 255, 0),
                    5: (255, 0, 0), 6: (128, 128, 128), 7: (186, 85, 211), 8: (255, 97, 3), 9: (0, 245, 255)}
# brown, #blue , #green, #pink, #yellow, #red, #grey, #purple, #orange, #turqize
dict_next_colors = {9: (156, 102, 31), 0: (0, 0, 255), 1: (127, 255, 0), 2: (255, 20, 147), 3: (255, 255, 0),
                    4: (255, 0, 0), 5: (128, 128, 128), 6: (186, 85, 211), 7: (255, 97, 3), 8: (0, 245, 255)}


class RainbowCurrMNIST(MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        super(RainbowCurrMNIST, self).__init__(root, train=train, download=download, transform=transform,
                                               target_transform=target_transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = image.numpy()
        color = dict_curr_colors[label.item()]
        image = np.stack((image, image, image), axis=2)
        image = np.where(image > 0, color, image)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class RainbowNextMNIST(MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        super(RainbowNextMNIST, self).__init__(root, train=train, download=download, transform=transform,
                                               target_transform=target_transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = image.numpy()
        color = dict_next_colors[label.item()]
        image = np.stack((image, image, image), axis=2)
        image = np.where(image > 0, color, image)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


transform = transforms.Compose([transforms.ToTensor()])
curr_colored_mnist_test = RainbowCurrMNIST(root='./data', train=False, download=True, transform=transform)
curr_data_loader = torch.utils.data.DataLoader(dataset=curr_colored_mnist_test, batch_size=120, shuffle=True,
                                               num_workers=4, pin_memory=True)
# test_loader_A = torch.utils.data.DataLoader(dataset=colored_mnistA_test, batch_size=120, shuffle=False, num_workers = 4)

next_colored_mnist_test = RainbowNextMNIST(root='./data', train=False, download=True, transform=transform)
next_data_loader = torch.utils.data.DataLoader(dataset=next_colored_mnist_test, batch_size=120, shuffle=True,
                                               num_workers=4, pin_memory=True)


# test_loader_B = torch.utils.data.DataLoader(dataset=colored_mnistB_test, batch_size=120, shuffle=False, num_workers = 4, pin_memory=True)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class D_curr(nn.Module):
    def __init__(self):
        super(D_curr, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sig(x)
        return x


class D_next(nn.Module):
    def __init__(self):
        super(D_next, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sig(x)
        return x


class G_C2N(nn.Module):
    def __init__(self):
        super(G_C2N, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class G_N2C(nn.Module):
    def __init__(self):
        super(G_N2C, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def generate(curr, next, c2n, n2c):
    c2n.eval()
    n2c.eval()

    next_fake = c2n(curr)
    curr_fake = n2c(next)

    curr_imgs = torch.zeros((curr.shape[0] * 2, 3, curr.shape[2], curr.shape[3]))
    next_imgs = torch.zeros((next.shape[0] * 2, 3, next.shape[2], next.shape[3]))

    even_idx = torch.arange(start=0, end=curr.shape[0] * 2, step=2)
    odd_idx = torch.arange(start=1, end=curr.shape[0] * 2, step=2)

    curr_imgs[even_idx] = curr.cpu()
    curr_imgs[odd_idx] = next_fake.cpu()

    next_imgs[even_idx] = next.cpu()
    next_imgs[odd_idx] = curr_fake.cpu()

    rows = math.ceil((curr.shape[0] * 2) ** 0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    a_imgs_ = torchvision.utils.make_grid(curr_imgs, nrow=rows).permute(1, 2, 0).numpy() * 255
    a_imgs_ = a_imgs_.astype(np.uint8)
    ax1.imshow(Image.fromarray(a_imgs_))
    ax1.set_xticks([])
    ax1.set_yticks([])

    b_imgs_ = torchvision.utils.make_grid(next_imgs, nrow=rows).permute(1, 2, 0).numpy() * 255
    b_imgs_ = b_imgs_.astype(np.uint8)
    ax2.imshow(Image.fromarray(b_imgs_))
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()


class pickle_cpu(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=lambda storage, loc: storage)
        else:
            return super().find_class(module, name)


if __name__ == '__main__':
    test_curr, _ = next(iter(curr_data_loader))
    test_next, _ = next(iter(next_data_loader))
    test_curr = to_cuda(test_curr)
    test_next = to_cuda(test_next)

    G_C2N = G_C2N()
    G_N2C = G_N2C()
    G_C2N.load_state_dict(torch.load('G_C2N_q2.pkl', map_location=lambda storage, loc: storage))
    G_N2C.load_state_dict(torch.load('G_N2C_q2.pkl', map_location=lambda storage, loc: storage))

    generate(test_curr, test_next, G_C2N, G_N2C)
