import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import librosa

class SpectrogramDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, augment=False):
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.npy')))
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))
        self.augment = augment

        # Expand dataset: one clean file paired with each noisy variation
        self.dataset_pairs = [
            (clean_file, noisy_file)
            for clean_file in self.clean_files
            for noisy_file in self.noisy_files
            if os.path.basename(clean_file).replace('_clean.npy', '') in noisy_file
        ]

        print(f"Total clean files: {len(self.clean_files)}")
        print(f"Total noisy files: {len(self.noisy_files)}")
        print(f"Total dataset pairs: {len(self.dataset_pairs)}")

        if not self.dataset_pairs:
            raise ValueError("No valid clean-to-noisy pairs found. Check your data.")

    def __len__(self):
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.dataset_pairs[idx]
        clean = np.load(clean_path)
        noisy = np.load(noisy_path)

        if self.augment:
            noisy = noisy * np.random.uniform(0.8, 1.2)

        return {
            "input": torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),  # Shape: [1, height, width]
            "target": torch.tensor(clean, dtype=torch.float32).unsqueeze(0),  # Shape: [1, height, width]
        }


# Define the CNN Model
class EnhancedCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(EnhancedCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Convert .wav to .npy
def convert_wav_to_npy(input_dir, output_dir, target_shape=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))
    if not wav_files:
        raise ValueError(f"No .wav files found in {input_dir}.")
    
    for wav_file in wav_files:
        y, sr = librosa.load(wav_file, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        resized_spectrogram = np.resize(spectrogram, target_shape)
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.npy')
        np.save(output_file, resized_spectrogram)

# Training Loop
if __name__ == "__main__":
    clean_wav_dir = "preprocessed/clean_audio"
    noisy_wav_dir = "preprocessed/noisy_audio"
    clean_npy_dir = "preprocessed/clean_audio_npy"
    noisy_npy_dir = "preprocessed/noisy_audio_npy"

    print("Preprocessing clean audio...")
    convert_wav_to_npy(clean_wav_dir, clean_npy_dir)

    print("Preprocessing noisy audio...")
    convert_wav_to_npy(noisy_wav_dir, noisy_npy_dir)

    dataset = SpectrogramDataset(clean_npy_dir, noisy_npy_dir, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedCNN(input_channels=1, output_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "enhanced_cnn_denoising.pth")
