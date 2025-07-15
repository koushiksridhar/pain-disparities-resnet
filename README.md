# Pain Disparities ResNet Analysis

This repository contains code for training and evaluating a ResNet-18 model to predict WOMAC pain scores from knee X-ray images, with a focus on analyzing pain disparities across demographic groups following the methodology of Pierson et al.

## Project Overview

The project consists of two main components:
1. **Model Training**: A ResNet-18 based regression model for predicting WOMAC pain scores from knee X-ray images. The pain score can be replaced to use KOOS as well as long as the data also uses KOOS units. 
2. **Disparity Analysis**: Analysis of pain disparities across demographic groups (race, education, income) while controlling for osteoarthritis severity.

## Data Requirements

### Directory Structure
You need to set up the following directory structure:

```
base_data_path/
├── data_splits_path/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
├── full_data_path/
│   └── knee.npz
└── save_model_path/
    └── (model outputs will be saved here)
```

### Data Files

#### 1. Data Split CSVs (`train_data.csv`, `val_data.csv`, `test_data.csv`)
Each of these data split CSV files should contain the Patient ID and Pain Score. An observation is denoted as one X-ray at one timepoint.

The full data CSV files are those that include additional demographic variables as well. 

#### 2. Knee Images (`knee.npz`)
A NumPy compressed file containing:
- `x`: Knee X-ray images as a 3D array (samples × height × width)
- `y`: Corresponding image names as a list of strings

### Data Format Requirements
- Images should be grayscale and will be resized to 128×128 pixels
- Image names in the CSV files should match those in the knee.npz file

## Environment Setup
You can install the environment.yml provided. This may be a bit overkill in terms of packages and its possible that not everything was used. It is likely that a lot of these packages in the YML file were dependencies for other packages as well. However, the main packages that were used were:
```
torch torchvision pandas numpy scikit-learn scipy matplotlib monai scikit-image
```

### Configuration
Update the data paths in `config.py` to match your directory structure:

```python
# In config.py
BASE_DATA_PATH = './your_base_data_path'
DATA_SPLITS_PATH = os.path.join(BASE_DATA_PATH, './your_data_splits_path')
FULL_DATA_PATH = os.path.join(BASE_DATA_PATH, './your_full_data_path')
SAVE_MODEL_PATH = os.path.join(BASE_DATA_PATH, './your_save_model_path')
```

## Usage

### 1. Model Training and Evaluation

#### Train a new model:
```bash
python run.py --devices 0,1,2,3 --train true
```

#### Evaluate with existing model:
```bash
python run.py --devices 0,1,2,3 --train false
```

#### Command Line Arguments:
- `--devices`: GPU device IDs to use (comma-separated)
- `--train`: Whether to train the model ("true" or "false")

### 2. Disparity Analysis

After running the model training/evaluation, use the Jupyter n

The notebook performs:
- Mean pain score calculations across demographic groups
- Disparity analysis controlling for KLG severity
- Disparity analysis controlling for model predictions
- Comparison of disparity reduction between KLG and model predictions

## Model Architecture

The model uses a ResNet-18 architecture with the following modifications:
- Pretrained ResNet-18 backbone
- Last 12 convolutional layers are fine-tuned
- Final fully connected layer replaced with:
  - Linear layer (512 → 256)
  - ReLU activation
  - Dropout (0.3)
  - Linear layer (256 → 1) for regression output

## Training Configuration

Key training parameters (configurable in `config.py`):
- **Batch size**: 8
- **Learning rate**: 0.001
- **Epochs**: 30
- **Image size**: 128×128
- **Optimizer**: Adam
- **Loss function**: Mean Squared Error
- **Scheduler**: ReduceLROnPlateau

## Output Files

After training, the following files are generated in the save directory:
- `best_model.pth`: Best performing model weights
- `training_history.csv`: Training metrics for each epoch
- `results.csv`: Test set predictions with actual vs predicted WOMAC scores

## Disparity Analysis Methodology

The disparity analysis follows the approach of Pierson et al.:

1. **Raw Disparities**: Calculate mean pain differences across demographic groups
2. **Controlled Disparities**: Analyze disparities while controlling for:
   - KLG (Kellgren-Lawrence Grade) - clinical severity measure
   - Model predictions - ResNet-based severity measure
3. **Reduction Analysis**: Compare how much each control variable reduces the raw disparity

## File Descriptions

- `run.py`: Main training and evaluation script
- `model.py`: ResNet-18 model definition
- `dataset.py`: Data loading and preprocessing
- `config.py`: Configuration parameters
- `utils.py`: Utility functions for image processing and device setup
- `disparity_analysis.ipynb`: Jupyter notebook for disparity analysis

## Notes

- Images are automatically flipped for right knee images to standardize orientation
- The training process includes validation and saves the best model based on validation loss
- All training metrics (RMSE, MAE, R², Pearson correlation) are logged and saved

