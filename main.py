import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from skimage import color
import numpy as np

from models import *
from losses import *
from utils import *

torch.manual_seed(0)

WEIGHTS_PATHS = "/content/drive/MyDrive/CycleGAN_weights/"

PRETRAINED = True
if PRETRAINED:
    LAST_STEP = get_last_training_step(WEIGHTS_PATHS)
if not LAST_STEP:
    PRETRAINED = False    


# Criterions
adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

# Training Params
n_epochs = 100
dim_A = 3
dim_B = 3
display_step = 2000
batch_size = 1
lr = 2e-4
load_shape = 286
target_shape = 256
device = "cuda"

# Data
transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

dataset = ImageDataset("horse2zebra", transform=transform)

# Models
gen_AB = Generator(dim_A, dim_B).to(device)
gen_BA = Generator(dim_B, dim_A).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Dicriminator(dim_A).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Dicriminator(dim_B).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constan_(m.bias, 0)


if PRETRAINED:
    pre_dict = torch.load(f"{WEIGHTS_PATHS}cycleGAN_{LAST_STEP}.pth")
    gen_AB.load_state_dict(pre_dict['gen_AB'])
    gen_BA.load_state_dict(pre_dict['gen_BA'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    disc_A.load_state_dict(pre_dict['disc_A'])
    disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
    disc_B.load_state_dict(pre_dict['disc_B'])
    disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
else:
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)


# Train

plt.rcParams["figure.figsize"] = (10, 10)

def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminatir_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        for real_A, real_B in tqdm(dataloader):
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            ### Update Discriminator A ###
            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()

            ### Update Discriminator B ###
            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()

            ### Update Generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminatir_loss += disc_A_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0:
                print(f"Epoch: {epoch}  Step: {cur_step + LAST_STEP}  GenLoss {mean_generator_loss}  DiscLoss: {mean_discriminatir_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminatir_loss = 0

                if save_model:
                    torch.save({
                       'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"{WEIGHTS_PATHS}cycleGAN_{cur_step + LAST_STEP}.pth")
            cur_step += 1

if __name__ == "__main__":
    train(save_model=True)
 
