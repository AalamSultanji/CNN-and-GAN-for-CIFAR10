import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.utils import save_image
import os
import sys

os.makedirs("outputs", exist_ok=True)

# hyper-parameters configurations
image_size = 32 # we need to resize image to 32X32
batch_size = 128
nz = 100 # latent vector size
beta1 = 0.5 # beta1 value for Adam optimizer
lr = 0.0001 # learning rate
sample_size = 32 # fixed sample size
epochs = 30 # number of epoch to train
num_classes = 10 # number of classes for cGAN
con_dim = 10 # dimension of the condition vector
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# image transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)),
])

# dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)


# generator
class Generator(nn.Module):
    def __init__(self, nz, num_classes=10, con_dim=10):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.con_dim = con_dim
        self.embed = nn.Linear(num_classes, con_dim)

        self.net = nn.Sequential(
            # Layer 1: nz+num_classes × 1 × 1 → 512 × 4 × 4
            nn.ConvTranspose2d(nz+con_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Layer 2: 512 × 4 × 4 → 256 × 8 × 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Layer 3: 256 × 8 × 8 → 128 × 16 × 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 4: 128 × 16 × 16 → 3 × 32 × 32
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        batch_size = z.size(0)
        z=z.view(batch_size, self.nz, 1, 1)
        one_hot=F.one_hot(labels, num_classes=self.num_classes).float()
        con_vec=self.embed(one_hot).view(batch_size, self.con_dim, 1, 1)
        x=torch.cat([z, con_vec], dim=1)
        return self.net(x)




# discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, con_dim=10, img_size=32):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.con_dim = con_dim
        self.embed = nn.Linear(num_classes, con_dim)
        self.img_size = img_size

        self.net = nn.Sequential(
            # Layer 1: 3 × 32 × 32 → 64 × 16 × 16
            nn.Conv2d(3+con_dim, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 64 × 16 × 16 → 128 × 8 × 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 128 × 8 × 8 → 256 × 4 × 4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 256 × 4 × 4 → 1 × 1 × 1
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        batch_size = x.size(0)
        one_hot=F.one_hot(labels, num_classes=self.num_classes).float()
        labels_map=self.embed(one_hot)
        labels_embed = labels_map.view(batch_size, self.con_dim, 1, 1)
        labels_embed = labels_embed.expand(-1, -1, self.img_size, self.img_size)
        x = torch.cat([x, labels_embed], dim=1)
        return self.net(x)



# train the network

# initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)



# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# loss function
criterion = nn.BCELoss()

# to store the generator and discriminator loss after each epoch
losses_g = []
losses_d = []

# creat the noise signal as input to the generator
def create_noise(sample_size,nz):
  return torch.randn(sample_size,nz,1,1).to(device)

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    # get the real label vector
    real_label = torch.ones(b_size, 1).to(device).squeeze()
    # get the fake label vector
    fake_label = torch.zeros(b_size, 1).to(device).squeeze()
    optimizer.zero_grad()
    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real,labels).view(-1)
    #print(output_real.size())
    loss_real = criterion(output_real, real_label)
    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake,labels).view(-1)
    loss_fake = criterion(output_fake, fake_label)
    # compute gradients of real loss
    loss_real.backward()
    # compute gradients of fake loss
    loss_fake.backward()
    # update discriminator parameters
    optimizer.step()
    return loss_real + loss_fake


# function to train the generator network
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    # get the real label vector
    real_label = torch.ones(b_size, 1).to(device).squeeze()
    optimizer.zero_grad()
    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake,labels).view(-1)
    loss = criterion(output, real_label)
    # compute gradients of loss
    loss.backward()
    # update generator parameters
    optimizer.step()
    return loss

# create the noise vector
noise = create_noise(sample_size, nz)

#sanity test
generator.eval()
discriminator.eval()

with torch.no_grad():
    z = create_noise(8, nz)
    labels = torch.randint(0, num_classes, (8,), device=device)

    fake = generator(z, labels)
    out = discriminator(fake, labels)

    print("Fake image shape:", fake.shape)
    print("Discriminator output shape:", out.shape)

generator.train()
discriminator.train()

#training loop

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for batch_idx, data in enumerate(trainloader):
        image, labels= data
        labels = labels.to(device)
        image = image.to(device)
        b_size = len(image)
        # forward pass through generator to create fake data
        z= create_noise(b_size, nz)
        data_fake = generator(z, labels).detach()
        data_real = image
        loss_d += train_discriminator(optim_d, data_real, data_fake)
        z= create_noise(b_size, nz)
        data_fake = generator(z, labels)
        loss_g += train_generator(optim_g, data_fake)
    # final forward pass through generator to create fake data...
    # ...after training for current epoch
    fixed_labels=torch.randint(0, num_classes, (sample_size,)).to(device)
    generated_img = generator(noise,fixed_labels).cpu().detach()
    # save the generated torch tensor models to disk
    save_image(generated_img, f"outputs/gen_img{epoch}.png", normalize=True)
    epoch_loss_g = loss_g / batch_idx # total generator loss for the epoch
    epoch_loss_d = loss_d / batch_idx # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    print(f"Epoch {epoch+1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
