# 🐶🐱 Dogs vs Cats — CNN + Transfer Learning
> Three-stage image classification: from a scratch-built CNN to VGG16 fine-tuning.

---

## Overview

A computer vision project that systematically compares three approaches to binary image classification — building up from a basic CNN to full transfer learning with VGG16. The progression is intentional: each stage fixes a specific problem from the previous one, making the reasoning visible and not just the result.

---

## Dataset

**Kaggle Dogs vs Cats** — 25,000 labelled images (12,500 dogs, 12,500 cats)  
Downloaded directly via Kaggle API inside the notebook.

- Balanced classes — no sampling needed
- High variety in pose, lighting, background, and breed
- Images resized to **224×224** (VGG16 standard — used across all models for a fair comparison)

---

## Approach — 3 Stages

### Stage 1 — Baseline CNN (from scratch)
- 3 blocks of Conv2D + MaxPooling
- No regularisation, no augmentation
- **Purpose:** Establish baseline accuracy and expose the overfitting problem

### Stage 2 — Improved CNN
Fixes every identified weakness from Stage 1:
- **BatchNormalization** after each conv block — stabilises training, speeds convergence
- **Dropout (0.4)** in the dense layers — combats overfitting
- **Data augmentation** (random horizontal flip, rotation ±10%, zoom ±10%) — increases effective training set diversity
- **EarlyStopping** + **ReduceLROnPlateau** — prevents wasted epochs, adapts learning rate automatically

### Stage 3 — VGG16 Transfer Learning
VGG16 was pretrained on 1.2 million ImageNet images. Its convolutional layers already encode low-level features (edges, textures) and mid-level features (shapes, patterns) that transfer well to new visual tasks.

**Phase 1 — Feature Extraction**
- Freeze all VGG16 layers
- Train only the new classification head (`GlobalAveragePooling2D` → `Dense(256)` → `Dropout(0.5)` → `Dense(1)`)
- Learning rate: `1e-4`

**Phase 2 — Fine-tuning**
- Unfreeze the last VGG16 conv block (block5)
- Retrain at a very low learning rate (`1e-5`) to adapt pretrained weights without destroying them
- EarlyStopping monitors validation loss

---

## Results

| Model | Val Accuracy |
|---|---|
| Baseline CNN | ~82% |
| Improved CNN (BatchNorm + Dropout + Augmentation) | ~87% |
| VGG16 Fine-tuned | **~95%+** |

**Training curves** for all three models are plotted side-by-side (accuracy and loss) so the improvement at each stage is visually clear.

---

## Visualisations

- Sample training image grid (8 images with labels)
- Training vs validation accuracy and loss curves — all 3 models overlaid
- Model comparison bar chart with accuracy labels
- Single image prediction with confidence score

---

## Tech Stack

```
Python · TensorFlow/Keras · OpenCV · pandas · matplotlib · numpy
```

---

## Run It

```bash
pip install tensorflow opencv-python matplotlib pandas numpy
```

1. Upload your `kaggle.json` API key to Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type) — mandatory for reasonable training time
3. Run all cells — dataset downloads automatically
4. Add your own test images as `/content/dog.jpg` and `/content/catt.jpg` for the prediction demo

---

## File Structure

```
CNN_Dogs_vs_Cats.ipynb         ← main notebook
dogs_vs_cats_vgg16.h5          ← saved best model (VGG16 fine-tuned)
```

---

## Why Transfer Learning Wins

Training a CNN from scratch on 25,000 images gives the model limited signal to learn robust visual features. VGG16 has already seen 1.2 million diverse images — it arrives with knowledge. Fine-tuning adapts that knowledge to the specific distribution of this dataset at a fraction of the compute cost. The ~13% accuracy gap over the improved custom CNN makes this one of the clearest demonstrations of why transfer learning is the default choice for most real-world image tasks.
