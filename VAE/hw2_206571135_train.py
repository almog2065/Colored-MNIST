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
import torchvision.utils as vutils
import matplotlib.pyplot as plt;plt.rcParams['figure.dpi'] = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class RainbowMNIST(MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        super(RainbowMNIST, self).__init__(root, train=train, download=download, transform=transform,
                                           target_transform=target_transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = image.numpy()
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = np.stack((image, image, image), axis=2)
        image = np.where(image > 0, color, image)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.z_dim = latent_dims
        self.linear1 = nn.Linear(28 * 28 * 3, 512)
        self.linear2 = nn.Linear(512, 256)
        self.to_mean_logvar = nn.Linear(256, 2 * latent_dims)

    def reparametrization_trick(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu, log_var = torch.split(self.to_mean_logvar(x), self.z_dim, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 2352)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))
        return z.reshape(-1, 3, 28, 28)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_con(vae, data, epochs=20):
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    for epoch in range(epochs):
        # running_loss = 0.0
        print(f'Epoch {epoch + 1}')
        # print("-"*40)
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = vae(x)
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + vae.encoder.kl
            # running_loss += loss.item()
            loss.backward()
            optimizer.step()
        # print(f"Epoch loss: {running_loss} \n")

    return vae


def plot_latent(autoencoder, data, num_batches=120):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

    plt.colorbar()


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = []
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            x_hat = autoencoder.decoder(z)
            img.append(x_hat)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=12).permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(img, extent=[*r0, *r1])


# dis


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-20):
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    bs, N, K = logits.size()
    y_soft = gumbel_softmax_sample(logits.view(bs * N, K), tau=tau, eps=eps)

    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=K)

        # 1. makes the output value exactly one-hot
        # 2.makes the gradient equal to y_soft gradient
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y.reshape(bs, N * K)


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(DiscreteVAE, self).__init__()

        self.fc1 = nn.Linear(28 * 28 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 2352)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim
        self.K = categorical_dim

    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        out = self.sigmoid(self.fc6(h5))
        return out.view(-1, 3, 28, 28)

    def forward(self, x, temp, hard):
        q = self.encoder(x.view(-1, 2352))
        q_y = q.view(q.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train_dis(train_loader, num_epochs=30, temp=1.0, hard=False):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    model = DiscreteVAE(3, 20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    train_loss = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        print("-" * 40)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, qy = model(x, temp, hard)
            loss = loss_function(x_hat, x, qy)
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        print(f"Epochs loss: {train_loss} \n")
        train_loss = 0.0

    return model


# joint

class JointVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(JointVAE, self).__init__()
        self.discrete = DiscreteVAE(latent_dim, categorical_dim)
        self.continous = VariationalAutoencoder(latent_dim)
        self.N = latent_dim
        self.K = categorical_dim

        self.fc4 = nn.Linear(latent_dim + latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 2352)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim
        self.K = categorical_dim

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        out = self.sigmoid(self.fc6(h5))
        return out.view(-1, 3, 28, 28)

    def forward(self, x, temp, hard):
        q = self.discrete.encoder(x.view(-1, 2352))
        q_y = q.view(q.size(0), self.N, self.K)
        z_dis = gumbel_softmax(q_y, temp, hard)
        qy = F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)

        z_con = torch.flatten(x, start_dim=1)
        z_con = F.relu(self.continous.encoder.linear1(z_con))
        z_con = F.relu(self.continous.encoder.linear2(z_con))
        mu, log_var = torch.split(self.continous.encoder.to_mean_logvar(z_con), self.N, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        z_con = self.continous.encoder.reparametrization_trick(mu, log_var)

        z = torch.concat((z_dis, z_con), dim=-1)
        return self.decoder(z), qy, self.kl


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train_jo(train_loader, num_epochs=20, temp=1.0, hard=False):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    model = JointVAE(3, 20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    train_loss = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        print("-" * 40)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, qy, kl = model(x, temp, hard)
            loss = loss_function(x_hat, x, qy) + kl
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        print(f"Epochs loss: {train_loss} \n")
        train_loss = 0.0

    return model


def main():
    # gaussian VAE
    z_dim = 2
    vae = VariationalAutoencoder(z_dim).to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    colored_mnist_train = RainbowMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=colored_mnist_train, batch_size=120, shuffle=True, num_workers=4)

    vae = train_con(vae, train_loader, epochs=30)

    plot_latent(vae, train_loader)
    plt.show()

    with open("gaussian_vae_q1.pkl", "wb") as f:
        pickle.dump(vae, f)

    plot_reconstructed(vae, r0=(-4, -3), r1=(1, 2), n=12)
    plt.show()

    plot_reconstructed(vae, r0=(1, 2), r1=(0, -1), n=12)
    plt.show()

    plot_reconstructed(vae, r0=(-1, -2.5), r1=(0.5, -2.5), n=12)
    plt.show()

    plot_reconstructed(vae, r0=(0, 1), r1=(0, 0.5), n=12)
    plt.show()

    # discrete VAE
    N = 3
    K = 20  # one-of-K vector
    model = train_dis(train_loader, 30)
    with open("discrete_vae_q1.pkl", "wb") as f:
        pickle.dump(model, f)

    ind = torch.zeros(N, 1).long()
    images_list = []
    for k in range(K):
        to_generate = torch.zeros(K * K, N, K)
        index = 0
        for i in range(K):
            for j in range(K):
                ind[1] = k
                ind[0] = i
                ind[2] = j
                z = F.one_hot(ind, num_classes=K).squeeze(1)
                to_generate[index] = z
                index += 1

        generate = to_generate.view(-1, K * N).to(device)
        reconst_images = model.decoder(generate)
        reconst_images = reconst_images.view(reconst_images.size(0), 3, 28, 28).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_images, nrow=K).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img))

        for i in range(len(images_list)):
            plt.imshow(images_list[i])
            plt.show()

    # joint VAE
    model = train_jo(train_loader, 50)
    with open("joint_vae_q1.pkl", "wb") as f:
        pickle.dump(model, f)
    ind = torch.zeros(N, 1).long()
    images_list = []
    for k in range(K):
        to_generate = torch.zeros(K * K, N, K)
        index = 0
        for i in range(K):
            for j in range(K):
                ind[1] = k
                ind[0] = i
                ind[2] = j
                z = F.one_hot(ind, num_classes=K).squeeze(1)
                to_generate[index] = z
                index += 1

        generate = to_generate.view(-1, K * N).to(device)
        z_con = torch.normal(0, 5, size=(generate.shape[0], N)).to(device)
        generate = torch.cat((generate, z_con), dim=-1)
        reconst_images = model.decoder(generate)
        reconst_images = reconst_images.view(reconst_images.size(0), 3, 28, 28).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_images, nrow=K).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img))

    for i in range(len(images_list)):
        plt.imshow(images_list[i])
        plt.show()


if __name__ == '__main__':
    main()
