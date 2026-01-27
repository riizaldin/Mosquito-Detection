# 🦟 Mosquito Species Detection Web App

AI-powered web application for detecting and classifying mosquito species using ensemble deep learning models.

## Features

- **Ensemble Model Architecture**: Combines three state-of-the-art models:
  - Swin Transformer Tiny (35% weight)
  - EfficientNetV2-S (35% weight)
  - ConvNeXt Tiny (30% weight)

- **Test Time Augmentation (TTA)**: Applies multiple transformations to improve prediction accuracy

- **Species Classification**: Detects three mosquito species:
  - Aedes aegypti (Yellow fever mosquito)
  - Aedes albopictus (Asian tiger mosquito)
  - Culex quinquefasciatus (Southern house mosquito)

- **Interactive Web Interface**: Built with Streamlit for easy use

## Requirements

- Python 3.8 or higher
- PyTorch with CUDA support (optional, for GPU acceleration)
- Streamlit
- Other dependencies listed in requirements.txt

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd d:\KULIAH\PROJECT\mosquitos_detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support, install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Ensure model files are present**:
   Make sure these files exist in the project directory:
   - `best_swin_t.pth`
   - `best_efficientnetv2_s.pth`
   - `best_convnext_t.pth`

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:
   - The app will automatically open in your default browser
   - Or navigate to: `http://localhost:8501`

3. **Use the application**:
   - Upload a mosquito image (JPG, JPEG, or PNG)
   - Toggle TTA (Test Time Augmentation) if desired
   - Click "Detect Mosquito Species"
   - View results with confidence scores and species information

## Model Architecture

### Ensemble Configuration
- **Swin Transformer Tiny**: Vision Transformer architecture (35% weight)
- **EfficientNetV2-S**: Efficient CNN architecture (35% weight)
- **ConvNeXt Tiny**: Modern CNN architecture (30% weight)

### Test Time Augmentation (TTA)
When enabled, applies 5 different transformations:
1. Original image
2. Horizontal flip
3. Rotation (±10 degrees)
4. Brightness adjustment
5. Contrast adjustment

Results are averaged for more robust predictions.

## Project Structure

```
mosquitos_detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── best_swin_t.pth                # Swin Transformer model weights
├── best_efficientnetv2_s.pth      # EfficientNetV2 model weights
├── best_convnext_t.pth            # ConvNeXt model weights
└── ensemble_training.ipynb        # Training notebook
```

## Mosquito Species Information

### Aedes aegypti (Yellow Fever Mosquito)
- **Danger Level**: High
- **Vector for**: Dengue, Zika, Yellow Fever, Chikungunya
- **Characteristics**: Black with white markings, lyre-shaped thorax pattern
- **Activity**: Daytime

### Aedes albopictus (Asian Tiger Mosquito)
- **Danger Level**: High
- **Vector for**: Dengue, Chikungunya, Yellow Fever
- **Characteristics**: Black with white stripes, single white stripe on thorax
- **Activity**: Day and early evening

### Culex quinquefasciatus (Southern House Mosquito)
- **Danger Level**: Medium
- **Vector for**: West Nile Virus, St. Louis Encephalitis, Zika
- **Characteristics**: Brown/tan color, no distinctive markings
- **Activity**: Nighttime

## Performance

The ensemble model achieves high accuracy through:
- Multiple model architectures capturing different features
- Weighted ensemble predictions
- Test Time Augmentation (optional)
- Transfer learning from ImageNet pre-trained weights

## Troubleshooting

### Model files not found
- Ensure all three .pth files are in the same directory as app.py
- Check file names match exactly: `best_swin_t.pth`, `best_efficientnetv2_s.pth`, `best_convnext_t.pth`

### Out of memory errors
- Disable TTA (reduces memory usage)
- Close other applications
- Use CPU mode if GPU memory is insufficient

### Slow predictions
- Enable GPU/CUDA support for faster inference
- Disable TTA for quicker results (slightly lower accuracy)

## Development

To retrain the models, use the `ensemble_training.ipynb` notebook with your own dataset.

## License

This project is for educational purposes.

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web framework
- Pre-trained models from torchvision
