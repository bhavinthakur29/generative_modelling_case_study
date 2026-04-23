# Generative Modelling – MSc Data Science Assignment

A Jupyter notebook exploring Generative Adversarial Networks (GANs) across three real-world domains.

## Structure

**Part 1 – GAN Fundamentals**  
Vanilla and DCGAN architectures trained on synthetic 2D distributions (sine wave, spiral, mixture of Gaussians, noisy parametric curve). Includes a side-by-side comparison of original vs. modified GAN architectures.

**Part 2 – Real-World Applications**

| Domain | Dataset | Model |
|---|---|---|
| Medicine | BloodMNIST (MedMNIST) | DCGAN |
| Cybersecurity | CIC-IDS-2017 network traffic | Tabular GAN (MLP) |
| Creative Arts | QuickDraw "Pizza" sketches | DCGAN |

## Key Techniques

- Label smoothing (real labels = 0.9) for discriminator regularisation
- `StandardScaler` normalisation for tabular features before tensor conversion
- FID score evaluation using 1 000 real vs. 1 000 synthetic samples
- Real vs. synthetic image grids (`torchvision.utils.make_grid`)
- Generator and discriminator loss curves saved per task
- t-SNE visualisation for tabular latent space comparison

## Requirements

```
torch torchvision medmnist pytorch-fid scikit-learn pandas numpy matplotlib opencv-python tqdm
```

```bash
pip install torch torchvision medmnist pytorch-fid scikit-learn pandas numpy matplotlib opencv-python tqdm
```

## Data

- **BloodMNIST** – downloaded automatically via `medmnist`
- **CIC-IDS-2017** – CSV files must be placed in the working directory and merged before training
- **QuickDraw Pizza** – `pizza.ndjson` must be placed in the working directory

## Outputs

Each task saves the following to the working directory:

```
bloodmnist_loss_curve.png
bloodmnist_real_vs_fake.png
cybersecurity_merged_loss_curve.png
cybersecurity_merged_real_vs_fake.png
cybersecurity_wednesday_loss_curve.png
pizza_loss_curve.png
pizza_real_vs_fake.png
```
