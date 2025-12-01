# Copy-Move Forgery Detection in Scientific Images

## Overview

This project implements a deep learning solution for detecting **copy-move forgeries** in biomedical and scientific images. Copy-move forgery is a type of image manipulation where regions of an image are duplicated and pasted elsewhere to fabricate results—one of the most common and damaging forms of scientific misconduct.

### The Problem

Every chart, figure, and microscope slide in a scientific paper tells a story. But today, that story may be fake. Copy-move forgeries can mislead researchers, waste time and funding, and undermine trust in entire fields of study.

Most detection still relies on expert reviewers manually scanning papers, or on digital forensics tools that weren't built with scientific figures in mind. Both approaches fall short in biomedical research:
- **Manual review** is slow and impossible to scale
- **Generic automated detectors** struggle with the complexity of scientific figures

This project addresses this gap by providing a tool built specifically for detecting copy-move forgeries in scientific images using state-of-the-art deep learning techniques.

## Dataset

The benchmark is based on several hundred confirmed forgeries pulled from more than 2,000 retracted papers, making it one of the most realistic and detailed datasets in scientific image forensics.

### Dataset Structure

```
data/
├── train_images/
│   ├── authentic/          # Authentic images (no forgeries)
│   │   └── *.png          # 2,377 images
│   └── forged/            # Images with copy-move forgeries
│       └── *.png          # 2,751 images
├── train_masks/
│   └── *.npy              # Binary masks for forged images (2,751 masks)
├── supplemental_images/
│   └── *.png              # Additional training images
├── supplemental_masks/
│   └── *.npy              # Masks for supplemental images
└── test_images/
    └── *.png              # Test images for inference
```

### Key Dataset Characteristics

- **Authentic Images**: Images with no forgeries (zero masks)
- **Forged Images**: Images containing copy-move forgeries (binary masks indicating forged regions)
- **Mask Format**: `.npy` files containing binary masks where `1` indicates forged regions and `0` indicates authentic regions
- **Multiple Instances**: Some masks may contain multiple forgery instances, which are merged using logical OR operation
- **Name Collisions**: Authentic and forged images may share the same filename but are in different folders—the notebook handles this correctly

## Model Architecture

### simplified UNet3Plus

The notebook implements **UNet3Plus**, an advanced encoder-decoder architecture for semantic segmentation:

- **Encoder**: Extracts hierarchical features using a series of convolutional blocks
- **Decoder**: Reconstructs the segmentation mask using full-scale skip connections
- **Full-Scale Skip Connections**: Aggregates features from all encoder levels, enabling better feature fusion
- **Output**: Single-channel binary mask indicating forged regions

### Architecture Details

- **Input**: Grayscale images converted to RGB (3 channels) at 256×256 resolution
- **Output**: Single-channel probability map (sigmoid activation) indicating forgery regions
- **Parameters**: Configurable encoder channels, skip connections, and dropout

## Training Approach

### Loss Function

The model uses a **combined loss function** that balances different aspects of segmentation:

1. **Tversky Loss** (weight: 0.7)
   - Asymmetric loss function that penalizes false negatives more than false positives
   - Parameters: `alpha=0.7`, `beta=0.3` (emphasizes recall for forgery detection)
   - Ideal for imbalanced datasets where forged regions are rare

2. **Binary Cross-Entropy (BCE) Loss** (weight: 0.3)
   - Standard pixel-wise classification loss
   - Provides stable gradients

**Combined Loss**: `L = 0.7 × Tversky + 0.3 × BCE`

### Metrics

The training process tracks multiple metrics:

- **Dice Coefficient**: Measures overlap between predicted and ground truth masks
  - Range: [0, 1], higher is better
  - Formula: `Dice = (2 × Intersection) / (Predicted + Ground Truth)`

- **IoU (Intersection over Union)**: Another overlap metric
  - Range: [0, 1], higher is better
  - Formula: `IoU = Intersection / Union`

### Training Configuration

- **Image Size**: 256×256 pixels
- **Batch Size**: 16 (configurable)
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 15%
- **Early Stopping**: Patience of 10 epochs
- **Device**: CUDA GPU (automatically detected)

#### Training Strategy

The notebook supports a **two-phase training approach**:

1. **Phase 1: Forged-Only Training (10 epochs)**
   - Trains exclusively on forged images (images with copy-move forgeries)
   - Helps the model learn to identify forgery patterns before seeing authentic examples
   - Focuses the model's attention on positive examples (forged regions)
   - Can be configured by filtering the dataset to only include forged images

2. **Phase 2: Full Dataset Training**
   - Trains on both authentic and forged images
   - Allows the model to learn the distinction between authentic and forged regions
   - Continues training for the remaining epochs (or until early stopping)


### Data Preprocessing

1.**Resizing**: All images resized to 256×256 using bilinear interpolation
2. **Normalization**: Images normalized to [0, 1] range using `torchvision.transforms.ToTensor()`
3. **Mask Processing**:
   - Multiple mask instances merged using logical OR operation
   - Masks resized to match image dimensions using nearest-neighbor interpolation
   - Binary thresholding at 0.5

### Handling Authentic vs. Forged Images

The notebook includes special handling for authentic images:

- **Authentic Images**: Always receive zero masks, even if a mask file exists with the same name (handles name collisions)
- **Forged Images**: Load corresponding mask files from `train_masks/` or `supplemental_masks/`
- **Path-Based Detection**: Uses folder path (`authentic` vs `forged`) to determine mask assignment



## Usage

### Prerequisites

```bash
pip install torch torchvision numpy pillow matplotlib tqdm kaggle
```

### Dataset Setup

1. **Install Kaggle API** (if not already installed):
   ```bash
   pip install kaggle
   ```

2. **Configure Kaggle API credentials**:
   - Go to your Kaggle account settings: https://www.kaggle.com/settings
   - Scroll down to "API" section and click "Create New Token"
   - This downloads `kaggle.json` file
   - Place `kaggle.json` in `~/.kaggle/` directory (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
   - Set proper permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac
     ```

3. **Download the dataset**:
   ```bash
   kaggle competitions download -c recodai-luc-scientific-image-forgery-detection
   ```

4. **Extract the dataset**:
   ```bash
   unzip recodai-luc-scientific-image-forgery-detection.zip -d data/
   ```
   
   Or on Windows:
   ```powershell
   Expand-Archive -Path recodai-luc-scientific-image-forgery-detection.zip -DestinationPath data/
   ```

5. **Verify dataset structure**: Ensure the `data/` folder contains the following structure:
   - `train_images/authentic/`
   - `train_images/forged/`
   - `train_masks/`
   - `supplemental_images/`
   - `supplemental_masks/`
   - `test_images/`

### Running the Notebook

1. **Open the Notebook**: `fraud_detection_Unet3Plus.ipynb`

2. **Configure Parameters** (Cell 5):
 

3. **Run Cells Sequentially**:
   - Cells 1-4: Imports and setup
   - Cell 5: Configuration
   - Cell 6: Dataset class definition
   - Cell 7: Model architecture
   - Cell 8: Loss functions and metrics
   - Cell 9: Dataset preparation
   - Cell 10: Helper functions
   - Cell 11: Model initialization
   - Cell 12: Training loop
   - Cell 13: Evaluation and inference

### Expected Output

- **Training Progress**: Loss, Dice, and IoU metrics printed after each epoch
- **Visualizations**: Images with mask overlays saved to `saved_models/`
- **Model Checkpoint**: Best model saved to `saved_models/best_model_full.pth`
- **Test Predictions**: Saved to `./output_masks/` directory

## To-Do List

### Model Architecture Enhancements

- [ ] **Deep Supervision**
  - [ ] Implement deep supervision at multiple decoder levels
  - [ ] Add auxiliary loss functions for intermediate decoder outputs
  - [ ] Weight the deep supervision losses appropriately
  - [ ] Update forward pass to return multiple outputs
  - [ ] Modify training loop to compute losses at each supervision level

- [ ] **Classification Guided Modules (CGM)**
  - [ ] Implement Classification Guided Module for UNet3Plus
  - [ ] Add image-level classification head (authentic vs. forged)
  - [ ] Integrate classification features into segmentation decoder
  - [ ] Add classification loss to training objective
  - [ ] Implement feature fusion between classification and segmentation branches

### Input Preprocessing Enhancements

- [ ] **SRM (Steganalysis Rich Model) Filters**
  - [ ] Implement SRM filter bank (2 high-pass filters)
  - [ ] Add SRM preprocessing layer to extract steganalysis features
  - [ ] Concatenate SRM features as additional input channels
  - [ ] Update model input channels to accommodate SRM features
  - [ ] Test impact on forgery detection performance

- [ ] **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
  - [ ] Implement CLAHE preprocessing
  - [ ] Add CLAHE as an additional input channel
  - [ ] Configure CLAHE parameters (clip limit, tile grid size)
  - [ ] Integrate CLAHE into data preprocessing pipeline
  - [ ] Evaluate CLAHE's effect on model performance

- [ ] **High-Pass Filters**
  - [ ] Implement high-pass filter bank (e.g., Sobel, Laplacian, Gaussian)
  - [ ] Extract edge and texture features using high-pass filters
  - [ ] Add high-pass filter outputs as additional input channels
  - [ ] Combine multiple high-pass filter responses
  - [ ] Update model architecture to handle multi-channel input

### Integration Tasks

- [ ] **Multi-Channel Input Architecture**
  - [ ] Modify UNet3Plus encoder to accept extended input channels
  - [ ] Update first convolutional layer to handle: RGB (3) + SRM (2) + CLAHE (1) + High-pass (N) channels
  - [ ] Test with different channel combinations
  - [ ] Optimize channel fusion strategy

- [ ] **Training Pipeline Updates**
  - [ ] Update dataset class to apply SRM, CLAHE, and high-pass filters
  - [ ] Modify data loading to include preprocessing steps
  - [ ] Update loss function to handle deep supervision outputs
  - [ ] Add classification loss to combined loss function
  - [ ] Update metrics to track classification accuracy alongside segmentation metrics

- [ ] **Competition F1 Loss Function**
  - [ ] Implement image-level F1 score calculation (competition metric)
  - [ ] Add F1 loss function for training optimization
  - [ ] Implement threshold and area fraction grid search for calibration
  - [ ] Test F1 loss as primary or auxiliary loss function
  - [ ] Compare F1 loss performance with Dice/Tversky losses
  - [ ] Integrate F1 loss into combined loss function
  - [ ] Validate F1 score calculation matches competition evaluation

- [ ] **CSV Output for Submission**
  - [ ] Implement RLE (Run-Length Encoding) for mask encoding
  - [ ] Create submission CSV with required format (image_id, mask_rle or "authentic")
  - [ ] Add image-level prediction logic (authentic vs. forged)
  - [ ] Implement threshold and area fraction calibration for binary decisions
  - [ ] Generate CSV output for test set predictions
  - [ ] Validate CSV format matches competition requirements
  - [ ] Add option to save predictions in both PNG/npy and CSV formats

- [ ] **Evaluation and Testing**
  - [ ] Benchmark performance with each enhancement individually
  - [ ] Test combined enhancements
  - [ ] Compare results with baseline model
  - [ ] Visualize feature maps from SRM and high-pass filters
  - [ ] Analyze computational overhead of additional preprocessing
  - [ ] Evaluate F1 score on validation set using competition metric
  - [ ] Test CSV generation on validation set before final submission




