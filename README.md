# Masked Autoencoder (MAE) Built From Scratch 🧩

A complete, ground-up implementation of a Masked Autoencoder (MAE) using pure PyTorch. This project demonstrates self-supervised learning by hiding 75% of an image's patches and forcing an asymmetric Vision Transformer to hallucinate the missing pixels.

The repository includes the model architecture, training notebook, and a real-time Streamlit web interface for interactive inference.

---

## 🎯 Project Overview
Standard Vision Transformers (ViTs) process 100% of an image. This Masked Autoencoder takes a more efficient and challenging approach:
1. **Patchify & Mask:** The input image is divided into a 14x14 grid of 16x16 pixel patches. 75% of these patches are randomly dropped.
2. **Encoder (ViT-Base):** Processes *only* the 25% visible patches, vastly reducing compute requirements.
3. **Decoder (ViT-Small):** Takes the encoded latent representations, inserts learnable `[MASK]` tokens for the missing patches, restores the original 2D spatial sequence, and reconstructs the image.

---

## 🔬 Ablation Study: L2 Regression vs. 3-Bit Color Classification
During development, I encountered "Mean Collapse" (where the model defaults to predicting the dataset's average muddy-grey color). To accelerate convergence on limited compute (Kaggle dual T4 GPUs on the TinyImageNet dataset), I conducted an ablation study.

I compared standard **L2 Pixel Regression** against the **"3-Bit Color" Classification Task** used for self-supervised pre-training by the authors of the original 2020 ViT paper. The 3-bit task forces the model to predict the mean color of a patch from a quantized 512-color palette instead of outputting continuous decimals.

### The Results
| Metric | L2 Pixel Regression (MSE) | 3-Bit Color Task (Cross-Entropy) |
| :--- | :--- | :--- |
| **Mean PSNR** | **19.19** | 15.93 |
| **Mean SSIM** | **0.61** | 0.46 |

<img width="426" height="157" alt="image" src="https://github.com/user-attachments/assets/dcae5e2d-4610-497d-a169-431c7e4fbc16" />
<img width="424" height="143" alt="image" src="https://github.com/user-attachments/assets/cec03868-dfbc-4726-a6ca-bfdda3356918" />


**Key Takeaways:**
* **Metric Bias:** PSNR is mathematically derived from Mean Squared Error (MSE). Because the L2 model was trained using MSE loss, it was inherently optimized to win this metric.
* **The SSIM Penalty:** Structural Similarity (SSIM) measures edges and textures. The 3-bit decoder fills missing patches with solid, flat blocks of color, destroying the internal "structure" of the patch and tanking the SSIM score.
* **Task Mismatch:** While 3-bit color prediction is excellent for *representation pre-training* on massive datasets (like the 300M images used by Google), continuous L2 regression proved to be the superior tool for actual *image restoration* on smaller datasets.

---

## 📁 Repository Structure

* `app.py`: The Streamlit web application for real-time inference and interactive masking.
* `mae-2.pth`: The trained model weights (managed via Git LFS due to its 430MB size).
* `self-supervised-img-learning-masked-autoencoder.ipynb`: The Jupyter/Kaggle notebook containing the training loop, multi-GPU `DataParallel` setup, and evaluation logic.
* `.gitattributes`: Configuration for Git Large File Storage (LFS).

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Git LFS installed on your machine to properly download the model weights:
```bash
git lfs install
```
## ⚙️ Installation

1. Clone the repository:
```bash
git clone [https://github.com/AhsanAkhlaq/MaskedAutoEncoder.git](https://github.com/AhsanAkhlaq/MaskedAutoEncoder.git)
cd MaskedAutoEncoder
```
2.Install the required Python packages:
```bash
pip install torch torchvision streamlit Pillow
```

## 🚀 Run the Web App

Launch the interactive Streamlit interface:
```bash
streamlit run app.py
```
Upload an image, adjust the masking ratio slider (from 10% to 90%), and watch the model reconstruct the missing pixels in real-time!

## 💡 Key Learnings

* **Tensor Manipulation:** Mastered sequence unshuffling using `torch.argsort()` and `torch.gather()` to rebuild 2D spatial grids from 1D latent sequences.
* **Normalization Math:** Solved visual artifacting by correctly tracking and reversing ImageNet mean/std normalization dynamically during evaluation.
* **Hardware Utilization:** Configured PyTorch `nn.DataParallel` and mixed-precision training (`torch.amp`) to maximize dual-GPU throughput.
