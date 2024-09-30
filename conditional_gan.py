# %%
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
DATA_DIR = r"C:\Users\Asus\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\datasets"
DATA_SET = "PTB XL"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


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
        # print(f"Metadata for idx {idx}: {metadata}")

        return signal, metadata

ecg_dataset = ECGDataset(df)

batch_size = 32
dataloader = DataLoader(ecg_dataset, batch_size=batch_size)




# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# %%
latent_dim = 100
signal_length = 1000
batch_size = 32

# %% [markdown]
# LINEAR 

# %%
# # import torch
# # import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self, noise_dim, metadata_dim, output_dim):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(noise_dim + metadata_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, output_dim),
#             nn.Tanh()  # Use Tanh to ensure the output is in a specific range (e.g., -1 to 1)
#         )

#     def forward(self, noise, metadata):
#         # Concatenate noise and metadata along the feature dimension
#         x = torch.cat([noise, metadata], dim=1)
#         return self.model(x)




# class Discriminator(nn.Module):
#     def __init__(self, signal_dim, metadata_dim):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(signal_dim + metadata_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, signals, metadata):
#         x = torch.cat([signals, metadata], dim=1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.sigmoid(x)
    


# %%
def save_signals_plot(real_signals, fake_signals, epoch, plot_save_path):
    num_channels = real_signals.shape[1]  # Assuming real_signals is of shape (batch_size, num_channels, signal_length)
    
    plt.figure(figsize=(20, 15))  # Adjust the figure size as needed
    
    for i in range(num_channels):
        plt.subplot(3, 4, i + 1)  # Create a 3x4 grid for 12 channels
        plt.plot(real_signals[0, i].cpu().detach().numpy(), label='Real Signal')
        plt.plot(fake_signals[0, i].cpu().detach().numpy(), label='Fake Signal', linestyle='--')
        plt.title(f'Channel {i + 1} - Epoch {epoch + 1}')
        plt.legend()
    
    plt.tight_layout()  # Adjust the spacing between plots
    plt.savefig(os.path.join(plot_save_path, f'signals_epoch_{epoch + 1}.png'))
    plt.close()


# %%
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, metadata_dim, num_channels, signal_length):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.metadata_dim = metadata_dim
        self.num_channels = num_channels
        self.signal_length = signal_length

        # Calculate the initial size
        self.initial_size = signal_length // 16

        self.fc = nn.Linear(noise_dim + metadata_dim, 256 * self.initial_size)
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, metadata):
        x = torch.cat([noise, metadata], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, self.initial_size)
        x = self.conv_layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_channels, signal_length, metadata_dim):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate the size of the flattened conv output
        self.conv_output_size = 256 * (signal_length // 16)
        
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size + metadata_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, signal, metadata):
        x = self.conv_layers(signal)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, metadata], dim=1)
        return self.fc(x)

# Example dimensions
batch_size = 32
noise_dim = 100
metadata_dim = 10
signal_length = 1000 
num_channels = 12  # Number of channels in ECG signal

# Instantiate the generator and discriminator
generator = Generator(noise_dim, metadata_dim, num_channels, signal_length)
discriminator = Discriminator(num_channels, signal_length, metadata_dim)

# Optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_signals = []
real_metadata = []

for batch in dataloader:
    signals, metadata = batch
    real_signals.append(signals)
    real_metadata.append(metadata)

real_signals = torch.cat(real_signals, dim=0).to(device)
real_metadata = torch.cat(real_metadata, dim=0).to(device)

print(f"Shape of real_signals: {real_signals.shape}")
print(f"Shape of real_metadata: {real_metadata.shape}")

# %%
batch_size = 32
noise_dim = 100
metadata_dim = 10
signal_length = 1000
num_channels = 12
num_epochs = 100
lr = 0.0002
beta1 = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the generator and discriminator
generator = Generator(noise_dim, metadata_dim, num_channels, signal_length).to(device)
discriminator = Discriminator(num_channels, signal_length, metadata_dim).to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

num_samples = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_signals = []
real_metadata = []

for batch in dataloader:
    signals, metadata = batch
    real_signals.append(signals)
    real_metadata.append(metadata)

real_signals = torch.cat(real_signals, dim=0).to(device)
real_metadata = torch.cat(real_metadata, dim=0).to(device)

print(f"Shape of real_signals: {real_signals.shape}")
print(f"Shape of real_metadata: {real_metadata.shape}")
dataset = TensorDataset(real_signals, real_metadata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Lists to store loss values for plotting
g_losses = []
d_losses = []

# Training loop
for epoch in range(num_epochs):
    for i, (real_signals, metadata) in enumerate(dataloader):
        batch_size = real_signals.size(0)
        real_signals = real_signals.to(device)
        metadata = metadata.to(device)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        g_optimizer.zero_grad()

        # Generate fake signals
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_signals = generator(noise, metadata)

        # Calculate generator loss
        g_loss = criterion(discriminator(fake_signals, metadata), real_labels)

        # Backpropagation and optimization
        g_loss.backward()
        g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()

        # Calculate discriminator loss on real signals
        real_loss = criterion(discriminator(real_signals, metadata), real_labels)

        # Calculate discriminator loss on fake signals
        fake_loss = criterion(discriminator(fake_signals.detach(), metadata), fake_labels)

        # Combine losses
        d_loss = (real_loss + fake_loss) / 2

        # Backpropagation and optimization
        d_loss.backward()
        d_optimizer.step()

        # Store losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os


def save_signals_plot(real_signals, fake_signals, epoch, save_path):
    # Assuming real_signals and fake_signals are tensors of shape (batch_size, 12, 1000)
    plt.figure(figsize=(15, 6))
    for i in range(12):  # Plotting all 12 leads
        plt.subplot(3, 4, i + 1)
        plt.plot(real_signals[0, i].cpu().numpy(), label='Real')
        plt.plot(fake_signals[0, i].detach().cpu().numpy(), label='Fake')
        plt.title(f'Lead {i + 1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'epoch_{epoch + 1}.png'))
    plt.close()

# Hyperparameters
batch_size = 32
noise_dim = 100
metadata_dim = 10
signal_length = 1000
num_channels = 12
num_epochs = 100
lr = 0.0002
beta1 = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the generator and discriminator
generator = Generator(noise_dim, metadata_dim, num_channels, signal_length).to(device)
discriminator = Discriminator(num_channels, signal_length, metadata_dim).to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

num_samples = 1000

real_signals = []
real_metadata = []

for batch in dataloader:
    signals, metadata = batch
    real_signals.append(signals)
    real_metadata.append(metadata)

real_signals = torch.cat(real_signals, dim=0).to(device)
real_metadata = torch.cat(real_metadata, dim=0).to(device)

print(f"Shape of real_signals: {real_signals.shape}")
print(f"Shape of real_metadata: {real_metadata.shape}")
dataset = TensorDataset(real_signals, real_metadata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Lists to store loss values for plotting
g_losses = []
d_losses = []

# Create a directory to save plots
save_path = 'ecg_plots'
os.makedirs(save_path, exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    for i, (real_signals, metadata) in enumerate(dataloader):
        batch_size = real_signals.size(0)
        real_signals = real_signals.to(device)
        metadata = metadata.to(device)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        g_optimizer.zero_grad()

        # Generate fake signals
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_signals = generator(noise, metadata)

        # Calculate generator loss
        g_loss = criterion(discriminator(fake_signals, metadata), real_labels)

        # Backpropagation and optimization
        g_loss.backward()
        g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()

        # Calculate discriminator loss on real signals
        real_loss = criterion(discriminator(real_signals, metadata), real_labels)

        # Calculate discriminator loss on fake signals
        fake_loss = criterion(discriminator(fake_signals.detach(), metadata), fake_labels)

        # Combine losses
        d_loss = (real_loss + fake_loss) / 2

        # Backpropagation and optimization
        d_loss.backward()
        d_optimizer.step()

        # Store losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    # Print progress and save plots every 10 epochs
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
        
        # Generate a batch of fake signals for plotting
        with torch.no_grad():
            sample_noise = torch.randn(batch_size, noise_dim).to(device)
            sample_metadata = torch.randn(batch_size, metadata_dim).to(device)
            sample_fake_signals = generator(sample_noise, sample_metadata)
        
        # Save plot
        save_signals_plot(real_signals, sample_fake_signals, epoch, save_path)

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig(os.path.join(save_path, 'training_losses.png'))
plt.close()

print("Training complete. Plots saved in", save_path)


