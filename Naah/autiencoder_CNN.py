import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
import librosa
import numpy as np
import torchaudio

class SpectrogramDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.npy')))
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))

        if len(self.clean_files) == 0 or len(self.noisy_files) == 0:
            raise ValueError(f"No .npy files found in {clean_dir} or {noisy_dir}.")

        # Verify each clean file has 3 corresponding noisy files
        expected_noisy_files = len(self.clean_files) * 3
        assert len(self.noisy_files) == expected_noisy_files, \
            f"Mismatch: {len(self.clean_files)} clean files and {len(self.noisy_files)} noisy files (expected {expected_noisy_files})."

        self.noisy_groups = [self.noisy_files[i:i+3] for i in range(0, len(self.noisy_files), 3)]

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean = np.load(self.clean_files[idx])
        noisy_group = [np.load(noisy_file) for noisy_file in self.noisy_groups[idx]]
        
        # Select one noisy file randomly from the group
        noisy = noisy_group[np.random.randint(len(noisy_group))]

        return {
            "input": torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
            "target": torch.tensor(clean, dtype=torch.float32).unsqueeze(0),
        }


        

# CNN Autoencoder Model
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, H/8, W/8)
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, H, W)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Training Function
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch["input"].to(device), batch["target"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss/len(val_loader):.4f}")



def convert_wav_to_npy(input_dir, output_dir, target_shape=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))

    for wav_file in wav_files:
        y, sr = librosa.load(wav_file, sr=None)  # Load audio file
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to dB scale
        resized_spectrogram = np.resize(spectrogram, target_shape)  # Resize to expected shape
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.npy')
        np.save(output_file, resized_spectrogram)




if __name__ == "__main__":
    # Convert .wav to .npy if necessary
    convert_wav_to_npy("preprocessed/clean_audio", "preprocessed/clean_audio_npy")
    convert_wav_to_npy("preprocessed/noisy_audio", "preprocessed/noisy_audio_npy")

    # Validate Conversion
    if not os.listdir("preprocessed/clean_audio_npy") or not os.listdir("preprocessed/noisy_audio_npy"):
        raise ValueError("Conversion from .wav to .npy failed. Ensure the directories contain the converted files.")

    # Paths
    clean_dir = "preprocessed/clean_audio_npy"
    noisy_dir = "preprocessed/noisy_audio_npy"

    dataset = SpectrogramDataset(clean_dir=clean_dir, noisy_dir=noisy_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Model, Loss, Optimizer
    model = CNNAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    train_model(model, train_loader, val_loader, epochs=50, criterion=criterion, optimizer=optimizer, device=device)

    # Save the Model
    torch.save(model.state_dict(), "cnn_autoencoder_denoising.pth")
