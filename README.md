# Flood Area Segmentation 🌊🛸

A production-ready deep learning pipeline for **binary semantic segmentation of flooded regions** using aerial/drone imagery. This repository implements multiple state-of-the-art segmentation architectures, data cleaning routines, advanced augmentations, and mixed-precision training.

---

## 🚀 Key Features

*   **Architectures:** U-Net (with EfficientNet-B3 & ResNet34 encoders) and DeepLabV3+ pre-trained on ImageNet.
*   **Robust Loss Function:** Combined **Dice Loss + Binary Cross-Entropy (BCE) Loss** to handle class imbalance (small flood regions in large landscapes).
*   **Training Enhancements:** 
    *   **Automatic Mixed Precision (AMP)** for faster training and reduced GPU memory usage.
    *   **Early Stopping** (patience = 7 epochs) to prevent overfitting.
    *   **Learning Rate Scheduling** (`ReduceLROnPlateau`) based on validation Dice score.
*   **Advanced Augmentations:** Albumentations pipeline featuring vertical/horizontal flips, random 90-degree rotations, brightness/contrast adjustments, and ImageNet normalization.

---

## 📊 Performance Metrics

The models are evaluated on multiple pixel-level metrics:
*   **Dice Coefficient** (F1-score)
*   **Intersection over Union (IoU)** / Jaccard Index
*   **Precision & Recall**

During training, the U-Net with the **EfficientNet-B3** backbone achieves:
*   **Validation Dice Score:** `~0.906`
*   **Validation IoU:** `~0.829`
*   **Validation Precision:** `~0.910`
*   **Validation Recall:** `~0.903`

---

## 📁 Repository Structure

```directory
├── Flood_Segmentation.ipynb    # Main pipeline notebook
├── .gitignore                  # Git exclusions (weights, datasets)
└── README.md                   # Project documentation
```

### Dataset Structure (Expected Local Directory)
To train the models locally, prepare your dataset in the following hierarchy:
```directory
Data/
├── Image/                      # Drone/aerial input images (.png, .jpg)
│   ├── image_001.png
│   └── ...
├── Mask/                       # Binary ground-truth masks (.png, .jpg)
│   ├── image_001.png
│   └── ...
└── metadata.csv                # Metadata mapping (optional)
```
*Note: The dataset cleaning script automatically scans and matches image-mask pairs using filename stems.*

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Ashutosh-2109/Flood-detection.git
    cd Flood-detection
    ```

2.  **Install Dependencies:**
    Make sure you have PyTorch installed (preferably with GPU/CUDA support). Then, run:
    ```bash
    pip install -r requirements.txt
    ```
    *Or install core libraries directly:*
    ```bash
    pip install segmentation-models-pytorch albumentations opencv-python pandas matplotlib tqdm
    ```

3.  **Run the Notebook:**
    Launch Jupyter and run `Flood_Segmentation.ipynb` to step through data cleaning, dataset creation, training, and inference.
    ```bash
    jupyter notebook Flood_Segmentation.ipynb
    ```

---

## 🧠 Models and Weights
The models save their best checkpoints locally as:
*   `best_model.pth` / `best_efficientnet_b3.pth`: U-Net with EfficientNet-B3 encoder
*   `best_resnet34.pth`: U-Net with ResNet34 encoder
*   `best_deeplab.pth`: DeepLabV3+ model

*Note: Since trained weight files (`.pth`) exceed the GitHub 100MB file size limit, they are ignored by Git. You can train the models from scratch using the provided notebook or host the weights on an external storage service (like Google Drive, HuggingFace, or AWS S3).*

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a PR or submit an issue.

## 📄 License
This project is licensed under the MIT License.
