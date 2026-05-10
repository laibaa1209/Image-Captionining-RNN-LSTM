# Image Captioning with Spatial Attention (V4)

This repository contains an optimized Image Captioning pipeline using EfficientNetB0 features and a vectorized LSTM decoder with Bahdanau Attention.

## Key Features
- **Spatial Features:** 7x7 grid extraction from EfficientNetB0 (total 49 regions).
- **Vectorized Decoding:** Fully parallel training using teacher forcing (10x faster than RNN loops).
- **Spatial Attention:** Dynamic attention maps visualizing where the model "looks".
- **Mixed Precision:** Uses `float16` for faster GPU performance on Kaggle/Colab.
- **Two-Phase Training:** Frozen embedding phase followed by global fine-tuning.

## Project Structure
- **Code/**: Jupyter Notebooks for training and experimentation.
- **Models/**: Pre-trained weights (.keras) and configuration files (.pkl).
- **Visualizations/**: Training plots, attention maps, and model dashboard.
- **Report/**: Formal technical project report.
- **predict.py**: Standalone script for captioning new images.

## Usage Instructions

### 1. Requirements
Ensure you have the following installed:
`tensorflow`, `numpy`, `pillow`, `pickle`

### 2. Running Inference (Test the Model)
To generate a caption for any image, use the standalone `predict.py` script:
```bash
python predict.py path/to/your/image.jpg
```
The script will load the spatial attention model from the `Models/` folder and print the generated caption to the terminal.

### 3. Training
To view or re-run the training process, open the notebook in the `Code/` directory:
- **Code/CNN_LSTM_Final.ipynb**
Ensure the dataset paths (Flickr8k) are correctly set for your environment (Kaggle/Local).

## Performance
- **Final Loss:** 2.62
- **BLEU-1:** 41.75%
- **Architecture:** EfficientNetB0 + Spatial Attention + Vectorized LSTM

