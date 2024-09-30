# %%
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
DATA_DIR = r"C:\Users\Asus\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\datasets"
DATA_SET = "PTB XL"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


parquet_file = os.path.join(DATA_DIR,DATA_SET,'ecg_data.parquet')
df = pd.read_parquet(parquet_file, engine='pyarrow')


class ECGDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        ecg_id = row['ecg_id']

        signal_data = row['signal']

        signal = np.concatenate([np.array(ch, dtype=np.float32) for ch in signal_data]).reshape(-1,1000)
        signal = torch.tensor(signal)

        metadata = {
            'age': torch.tensor(row['age'], dtype=torch.float32),
            'sex': torch.tensor(row['sex'], dtype=torch.float32),
            'height': torch.tensor(row['height'], dtype=torch.float32),
            'weight': torch.tensor(row['weight'], dtype=torch.float32),
            'NORM': torch.tensor(row['NORM'], dtype=torch.float32),
            'MI': torch.tensor(row['MI'], dtype=torch.float32),
            'STTC': torch.tensor(row['STTC'], dtype=torch.float32),
            'HYP': torch.tensor(row['HYP'], dtype=torch.float32),
            'CD': torch.tensor(row['CD'], dtype=torch.float32),
            'strat_fold': torch.tensor(row['strat_fold'], dtype=torch.float32),
        }

        return signal, metadata

ecg_dataset = ECGDataset(df)

batch_size = 32
dataloader = DataLoader(ecg_dataset, batch_size=batch_size)

# Iterate over the DataLoader
# for signals, metadata in dataloader:
    # print(signals)  # Should be (batch_size, signal_length)
    # print(metadata)  # Dictionary of metadata for the batch
    # print(signals.shape) --> 32,12,1000

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# %%
latent_dim = 100
signal_length = 1000
batch_size = 32

# %%
import torch.nn.functional as F

class SelfAttention1D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention1D, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)
        out = self.gamma * out + x
        return out, attention

class ECGGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=12, output_length=1000):
        super(ECGGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256 * 125)  # Upscale to a feature map*

        self.conv1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        
        # Self-Attention layer
        self.attention = SelfAttention1D(64)
        
        self.conv3 = nn.ConvTranspose1d(64, output_channels, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Assuming the ECG signals are normalized between -1 and 1

    def forward(self, z):
        x = self.fc1(z)
        x = x.view(-1, 256, 125)  # Reshape to (batch_size, 256, 125)
        x = self.relu(self.conv1(x))  # Output shape: (batch_size, 128, 250)
        x = self.relu(self.conv2(x))  # Output shape: (batch_size, 64, 500)
        
        # Apply self-attention
        x, _ = self.attention(x)
        
        x = self.tanh(self.conv3(x))  # Output shape: (batch_size, 12, 1000)
        return x


# %%
# import torch.nn as nn

class ECGDiscriminator(nn.Module):
    def __init__(self, input_channels=12, input_length=1000):
        super(ECGDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Final classification layer
        self.fc1 = nn.Linear(512 * 62, 1)  # After 4 conv layers, length is reduced to 62

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512 * 62)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

import torch.nn as nn



# %%
import torch
import torch.nn as nn

# Instantiate the Generator and Discriminator

latent_dim = 100  # Size of the noise vector
generator = ECGGenerator(latent_dim=latent_dim)
discriminator = ECGDiscriminator()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = generator.to(device)
discriminator = discriminator.to(device)

# Print the models to verify the architecture
# print(generator)
# print(discriminator)


# %%
def save_signals_plot(real_signals, generated_signals, epoch, num_signals=5, save_path='plots/'):
    plt.figure(figsize=(15, 10))
    for i in range(num_signals):
        plt.subplot(num_signals, 2, 2*i+1)
        plt.plot(real_signals[i].cpu().detach().numpy(), color='blue')
        plt.title('Real ECG')
        
        plt.subplot(num_signals, 2, 2*i+2)
        plt.plot(generated_signals[i].cpu().detach().numpy(), color='red')
        plt.title('Generated ECG')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'{save_path}epoch_{epoch}.png')
    plt.close()
    

# %%
import matplotlib.pyplot as plt

def save_signals_plot(real_signals, fake_signals, epoch, save_path='plots/'):
    num_channels = real_signals.size(1)
    channels_per_plot = 6
    num_plots = num_channels // channels_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * channels_per_plot
        end_idx = start_idx + channels_per_plot

        fig, axs = plt.subplots(2, channels_per_plot, figsize=(20, 5))
        for i in range(channels_per_plot):
            channel_idx = start_idx + i
            axs[0, i].plot(real_signals[0, channel_idx].cpu().detach().numpy())
            axs[0, i].set_title(f'Real Channel {channel_idx + 1}')
            axs[1, i].plot(fake_signals[0, channel_idx].cpu().detach().numpy())
            axs[1, i].set_title(f'Fake Channel {channel_idx + 1}')

        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch+1}_part{plot_idx+1}.png')
        plt.close()
import torch
import torch.nn.functional as F

# Gradient penalty function
def compute_gradient_penalty(discriminator, real_samples, fake_samples, lambda_gp=10):
    """
    Computes the gradient penalty for WGAN-GP.

    Args:
        discriminator (nn.Module): The discriminator model.
        real_samples (torch.Tensor): Real data samples.
        fake_samples (torch.Tensor): Fake data samples generated by the generator.
        lambda_gp (float): The gradient penalty coefficient.

    Returns:
        torch.Tensor: The gradient penalty term.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1).to(real_samples.device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    # Forward pass of interpolated data through the discriminator
    d_interpolates = discriminator(interpolates)

    # Compute gradients w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute the gradient norm
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# Discriminator loss with gradient penalty
def discriminator_loss_with_gp(discriminator, real_samples, fake_samples, lambda_gp=10):
    """
    Computes the discriminator loss with gradient penalty.

    Args:
        discriminator (nn.Module): The discriminator model.
        real_samples (torch.Tensor): Real data samples.
        fake_samples (torch.Tensor): Fake data samples generated by the generator.
        lambda_gp (float): The gradient penalty coefficient.

    Returns:
        torch.Tensor: The total loss for the discriminator including the gradient penalty.
    """
    # Real and fake predictions
    real_preds = discriminator(real_samples)
    fake_preds = discriminator(fake_samples)

    # Wasserstein loss
    loss_real = -torch.mean(real_preds)
    loss_fake = torch.mean(fake_preds)
    wass_loss = loss_real + loss_fake

    # Compute gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples, lambda_gp)

    # Total discriminator loss
    d_loss = wass_loss + gradient_penalty
    return d_loss

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Integrate the save function into the training loop
def train_gan(generator, discriminator, dataloader, num_epochs=100, latent_dim=100, lr=0.0002, betas=(0.5, 0.999), save_interval=1, save_path='plots/'):
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=betas)
    lambda_gp = 5
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_signals, _) in enumerate(dataloader):
            real_signals = real_signals.to(device)
            batch_size = real_signals.size(0)

            # Labels for real and fake data
            real_labels = torch.ones(batch_size, 1)*0.9 
            fake_labels = torch.zeros(batch_size, 1)+0.1 
            real_labels = real_labels.to(device)
            fake_labels = fake_labels.to(device)
            # =======================
            # Train Discriminator
            # =======================
            optimizer_D.zero_grad()

            # Real signals
            outputs = discriminator(real_signals)
            d_loss_real = criterion(outputs, real_labels)

            # Fake signals
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_signals = generator(z)
            outputs = discriminator(fake_signals.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # # Total discriminator loss
            # d_loss = d_loss_real + d_loss_fake
            d_loss = discriminator_loss_with_gp(discriminator, real_signals, fake_signals, lambda_gp)
            d_loss.backward()
            optimizer_D.step()

            # =======================
            # Train Generator
            # =======================
            optimizer_G.zero_grad()

            # Generate fake signals and get discriminator's output
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_signals = generator(z)
            outputs = discriminator(fake_signals)

            # Generator wants to fool the discriminator
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

        # Save the generated signals every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_signals_plot(real_signals, fake_signals, epoch, save_path)

        # Print losses
        print(f'Epoch [{epoch+1}/{num_epochs}] | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}')


# %%
train_gan(generator, discriminator, dataloader, num_epochs=1000, latent_dim=100, lr=0.0002, betas=(0.5, 0.999), save_interval=1)



