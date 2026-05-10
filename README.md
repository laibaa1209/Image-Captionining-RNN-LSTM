# Image Captioning with Spatial Attention (V4)

This repository contains an optimized Image Captioning pipeline using EfficientNetB0 features and a vectorized LSTM decoder with Bahdanau Attention.

## Key Features
- **Spatial Features:** 7x7 grid extraction from EfficientNetB0 (total 49 regions).
- **Vectorized Decoding:** Fully parallel training using teacher forcing (10x faster than RNN loops).
- **Spatial Attention:** Dynamic attention maps visualizing where the model "looks".
- **Mixed Precision:** Uses `float16` for faster GPU performance on Kaggle/Colab.
- **Two-Phase Training:** Frozen embedding phase followed by global fine-tuning.

## Files
- `image_captioning_v4.py`: Main script/notebook code.
- `project_report.pdf`: Detailed academic analysis of the project.
- `dashboard_v4.png`: Training and evaluation summary dashboard.

## Usage Instructions

### 1. Requirements
Ensure you have the following installed:
- TensorFlow 2.10+
- NumPy, Pandas, Matplotlib, PIL
- NLTK (for BLEU evaluation)
- tqdm

### 2. Dataset Setup (Kaggle)
This code is optimized for Kaggle. Ensure the following datasets are added:
- `adityajn105/flickr8k`
- `danielwillgeorge/glove6b200dpretrained`

### 3. Running the Script
The code is divided into logical cells. Run them in order:
1. **Cells 1-4:** Data loading and preprocessing.
2. **Cell 5:** Feature extraction (takes ~10 mins on GPU).
3. **Cells 6-8:** Model building and dataset creation.
4. **Cells 9-11:** Training (Phase 1 & Phase 2).
5. **Cell 12-14:** Saving the model and configurations.
6. **Cells 15-20:** Evaluation, Inference, and Visualizations.

### 4. Inference (Captioning a new image)
To caption your own images, use the `beam_caption` function:
```python
feat = extract_feature("path_to_your_image.jpg")
caption = beam_caption(model_v4, feat, word_to_idx, idx_to_word, max_len)
print(caption)
```

## Performance
- **Final Loss:** 2.62
- **BLEU-1:** 41.75%
- **Speed:** ~60ms/step on Tesla P100.
