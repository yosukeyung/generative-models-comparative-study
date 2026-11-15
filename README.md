# Empirical Comparative Study of Generative Models & MLP Classifier Evaluation

This repository contains the PyTorch implementation for the Deep Learning Take-Home Assignment (D6421) Session 07. The project focuses on a full-scale empirical comparative study of six generative model architectures and one classification model using the **Fashion-MNIST** dataset.

## Implemented Models

1. MLP Classifier (Baseline & Theoretical Proof of Information Bottleneck)
2. Vanilla Autoencoder (AE)
3. Sparse Autoencoder with $L_1$ Sparsity Penalty
4. Variational Autoencoder (VAE)
5. Vanilla Generative Adversarial Network (GAN)
6. Deep Convolutional GAN (DCGAN)
7. Denoising Diffusion Probabilistic Models (DDPM - Light U-Net)

## Evaluation Metrics

- **Reconstruction Fidelity:** Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM).
- **Generative Quality:** Fréchet Inception Distance (FID) Score.
- **Performance/Efficiency:** Training time per epoch, Inference Latency per batch.

## Prerequisites

Ensure you have Python 3.8+ installed. A GPU is highly recommended (e.g., NVIDIA T4 on Google Colab) to accelerate the training and inference processes.

## Installation and Execution

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yosukeyung/generative-models-comparative-study
   cd generative-models-comparative-study
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual environment before installing the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. **Running the Experiment:**
   This project is encapsulated in a Jupyter Notebook to facilitate step-by-step execution and visualization.
   - Open the `Tugas_7.ipynb` file using Jupyter Notebook, Jupyter Lab, or Google Colab.
   - Run all cells sequentially.
   - **Note for Google Colab Users:** If you encounter an error related to `torch-fidelity` during the FID Score calculation, please click **Restart Session** in the Runtime menu, then re-run the FID calculation cell.

## Highlights

- **Generative Trilemma Verified:** The empirical results demonstrate that VAE and DCGAN are extremely fast (latency < 0.05 ms) but sacrifice structural continuity or stability (mode collapse). Meanwhile, DDPM provides superior fidelity and mode coverage but suffers from high latency (262.20 ms) due to the 100-timestep reverse diffusion process.
- **Information Bottleneck:** It is empirically proven that the latent space of a classification model (MLP) cannot be used to synthesize or perfectly reconstruct images because it has irreversibly discarded intra-class spatial and texture features during training.

## Repository Structure

- `Tugas_7.ipynb`: The main experiment script (Data Loading, Training, Evaluation, Visualizations).
- `requirements.txt`: List of required Python libraries.
- `README.md`: Project documentation.
- `TUGAS_DL_SESI07_2802428066_Yosuke_Yung.pdf`: The comprehensive final analysis report.
