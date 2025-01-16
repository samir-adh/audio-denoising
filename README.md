# **audio-denoising**

This project implements audio denoising using deep learning models such as **U-Net**, **ResNet**, and **DeepLabV3+**, leveraging clean audio from **LibriSpeech** and noise samples from **ESC-50**.

---

## **Installation**

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

---

## **Data Preprocessing**

To preprocess the datasets and create noisy samples, run:

```bash
python scripts/create_data.py
```

Ensure your datasets are organized as follows:

```plaintext
datasets/
├── ESC-50-master/         # Noise samples (ESC-50 dataset)
├── LibriSpeech/           # Clean speech samples (LibriSpeech dataset)
```

This script will generate preprocessed files and save them in the `data/` folder:

```plaintext
data/
├── clean_frames.npy        # Clean audio frames
├── noisy_frames.npy        # Noisy audio frames
```

---

## **Training**

To train a model, choose one of the following scripts:

For ResNet-based model:
```bash
python scripts/model_ResNet.py
```

For DeepLabV3+ model:
```bash
python scripts/model_DeepLabV3.py
```

The best model weights will be saved in:

```plaintext
models/weights/
├── best_model.keras
├── model_DeepLabV3Plus_best.keras
├── model_ResNet.keras
```

---

## **Evaluation**

To evaluate a trained model, run:

```bash
python scripts/evaluate_model.py
```

This will output metrics such as loss and mean absolute error (MAE) for the test set.

---

## **Prediction**

To denoise a noisy audio file using a trained model, run:

```bash
python scripts/prediction.py
```

Provide the path to the noisy audio file, and the script will output the denoised version in the specified directory.

---

## **Folder Structure**

Here’s an overview of the project structure:

```plaintext
audio-denoising/
├── datasets/                  # Contains LibriSpeech and ESC-50 datasets
├── models/
│   ├── weights/               # Trained model weights
│   │   ├── best_model.keras
│   │   ├── model_DeepLabV3Plus_best.keras
│   │   ├── model_ResNet.keras
├── scripts/
│   ├── create_data.py         # Preprocess datasets
│   ├── evaluate_model.py      # Evaluate trained models
│   ├── model_DeepLabV3.py     # DeepLabV3+ implementation
│   ├── model_ResNet.py        # ResNet-based implementation
│   ├── model_ResNet50.py      # ResNet50-based implementation
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
└── data/
    ├── clean_frames.npy       # Preprocessed clean audio frames
    ├── noisy_frames.npy       # Preprocessed noisy audio frames
```

---

## **Dependencies**

All required Python libraries are listed in `requirements.txt`. You can install them with:

```bash
pip install -r requirements.txt
```

---

## **Models**

### **1. U-Net**
- A simple encoder-decoder architecture for denoising.

### **2. ResNet**
- Uses residual connections for enhanced noise learning.

### **3. DeepLabV3+**
- Advanced architecture designed for complex noise removal tasks.



