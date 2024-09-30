# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
DATA_DIR = r"C:\Users\Asus\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\datasets"
DATA_SET = "PTB XL"
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
#     print(signals)  # Should be (batch_size, signal_length)
    # print(metadata)  # Dictionary of metadata for the batch
    # print(signals.shape) --> 32,12,1000

# %%
latent_dim = 100
signal_length = 1000
batch_size = 32

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.bmm(q, k.transpose(-2, -1))
        attention = self.softmax(attention / (x.size(-1) ** 0.5))
        
        out = torch.bmm(attention, v)
        return out

class ECGGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=12, output_length=1000):
        super(ECGGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_channels * output_length)

        self.attention = SelfAttention(output_channels * output_length)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        x = self.tanh(x)
        x = x.view(-1, 12, 1000)
        return x

class ECGDiscriminator(nn.Module):
    def __init__(self, input_channels=12, input_length=1000):
        super(ECGDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_channels * input_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (real_ecgs, _) in enumerate(dataloader):
            batch_size = real_ecgs.size(0)
            real_ecgs = real_ecgs.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Real ECGs
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_ecgs)
            d_loss_real = criterion(real_output, real_labels)

            # Fake ECGs
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_ecgs = generator(z)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_ecgs)
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total Discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_ecgs = generator(z)
            fake_output = discriminator(fake_ecgs)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    return generator, discriminator


# %%
latent_dim = 100  # Dimension of the latent space (input to the generator)
output_channels = 12  # Number of output channels (e.g., 12-lead ECG)
output_length = 1000  # Length of the ECG signal
input_channels = output_channels  # Input channels for discriminator
input_length = output_length  # Length of the input to the discriminator
num_epochs = 50  # Number of epochs to train the GAN
batch_size = 32  # Batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = ECGGenerator(latent_dim=latent_dim, output_channels=output_channels, output_length=output_length)
discriminator = ECGDiscriminator(input_channels=input_channels, input_length=input_length)


# %%
import os
import matplotlib.pyplot as plt

def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device, save_path):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        for i, (real_ecgs, _) in enumerate(dataloader):
            batch_size = real_ecgs.size(0)
            real_ecgs = real_ecgs.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Real ECGs
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_ecgs)
            d_loss_real = criterion(real_output, real_labels)

            # Fake ECGs
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_ecgs = generator(z)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_ecgs)
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total Discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_ecgs = generator(z)
            fake_output = discriminator(fake_ecgs)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

        # Save the plot every epoch or at specific intervals
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            save_signals_plot(real_ecgs, fake_ecgs, epoch, save_path)

    return generator, discriminator

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


# %%

ecg_dataset = ECGDataset(df)
dataloader = DataLoader(ecg_dataset, batch_size=batch_size, shuffle=True)

# Initialize the models
generator = ECGGenerator(latent_dim=latent_dim, output_channels=output_channels, output_length=output_length).to(device)
discriminator = ECGDiscriminator(input_channels=input_channels, input_length=input_length).to(device)

# Train the GAN
save_path = './ecg_plots'
trained_generator, trained_discriminator = train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device, save_path=save_path)



