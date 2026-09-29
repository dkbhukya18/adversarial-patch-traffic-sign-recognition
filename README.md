# Adversarial Patch Attacks on Traffic Sign Recognition (RP2-style) and Defenses

Course project for CIS 582, University of Michigan-Dearborn, Winter 2024.
Team: Danijela Jovanovski, Mithali Deepak Kumar Singh, Aditya Kothari, Dileep Kumar Bhukya.

> **Status:** corrected and fully re-run (Sept 2026, Colab T4). All numbers below come from the fixed notebook; the original course report's numbers are superseded. See [ERRATA.md](ERRATA.md) for what changed and why.

## What the project does

Traffic sign recognition (TSR) models in driver-assistance systems can be fooled by small, sticker-like perturbations. This project:

1. Trains two classifier architectures on two datasets:
   - Architectures: a custom CNN and an ImageNet-pretrained ResNet18 (fine-tuned).
   - Datasets: GTSRB (German, 43 classes) and LISA (US, 47 classes).
2. Attacks every model with a **targeted, mask-constrained perturbation** based on Robust Physical Perturbations (RP2, Eykholt et al., CVPR 2018). Noise is optimized only inside a sticker-shaped mask so the sign is classified as a chosen target label (label 2: "50 km/h" for GTSRB, "curveRight" for LISA).
3. Evaluates two defenses:
   - **Data augmentation**: random rotation up to 5°, plus color jitter on brightness, contrast, saturation and hue.
   - **Adversarial training**: pre-generated attack images are mixed into every training step.

## Scope note: "RP2-style", not full RP2

Full RP2 optimizes the perturbation over many physical transformations (distance, angle, lighting), known as expectation over transformations, and validates it with printed stickers. This implementation performs a **digital** masked targeted attack on single images, with no transformation sampling and no physical tests. The results describe digital robustness only.

## Metrics

- **Clean test accuracy**: accuracy on the unmodified test set.
- **Accuracy under attack**: accuracy after the attack is applied to each test image, with a fresh white-box attack against each model.
- **Attack success rate (ASR)**: the fraction of test images whose true label is *not* the target that the model classifies *as* the target.

## Results

Clean test accuracy, accuracy on the test set under the targeted attack, and attack success rate (ASR), for standard training (std) and adversarial training (adv). Raw numbers: [`results.jsonl`](results.jsonl).

| Model | Clean acc. (std) | Under attack (std) | ASR (std) | Clean acc. (adv) | Under attack (adv) | ASR (adv) |
|---|---|---|---|---|---|---|
| GTSRB · CNN | 96.3% | 44.3% | 28.2% | 97.3% | **85.8%** | 2.9% |
| GTSRB · CNN + aug | 97.2% | 55.6% | 6.3% | 97.6% | **87.1%** | 2.3% |
| GTSRB · ResNet18 | 96.4% | 36.5% | 19.1% | 95.1% | **68.0%** | 4.2% |
| GTSRB · ResNet18 + aug | 96.0% | 32.9% | 25.1% | 96.5% | **71.8%** | 3.3% |
| LISA · CNN | 98.8% | 76.7% | 2.1% | 98.7% | **67.3%** | 0.9% |
| LISA · CNN + aug | 98.3% | 81.9% | 0.6% | 95.9% | **68.2%** | 0.9% |
| LISA · ResNet18 | 98.5% | 55.3% | 3.9% | 98.8% | **65.3%** | 1.5% |
| LISA · ResNet18 + aug | 97.3% | 39.0% | 0.9% | 98.8% | **62.6%** | 0.6% |

### Takeaways

- **Adversarial training works on GTSRB.** Accuracy under attack roughly doubles for every GTSRB model (e.g. CNN 44.3% → 85.8%, ResNet18 36.5% → 68.0%), and targeted success drops to 2–4%, with clean accuracy within about 1 point.
- **On LISA the picture is mixed.** Adversarial training helped both ResNets (55.3% → 65.3% and 39.0% → 62.6%) but *hurt* both CNNs (76.7% → 67.3% and 81.9% → 68.2%). LISA's training set is small (5,499 images), and the adversarial examples all use one fixed mask and target, so the CNNs appear to overfit to that pattern rather than become generally robust.
- **Data augmentation is not a reliable defense.** It helped the CNNs but made both ResNets *more* vulnerable to the attack (GTSRB 36.5% → 32.9%, LISA 55.3% → 39.0%).
- **The fine-tuned ResNets were more vulnerable than the small CNNs** under this attack on both datasets.
- **The original report's LISA augmentation results (99.4% and 100% clean accuracy) were most likely measured on training images**, since the code pointed at the training folder. On the real test set these models score 98.3% and 97.3%.

## Fixes in this version

A summary follows. [ERRATA.md](ERRATA.md) has the full details.
- The augmented LISA models are now evaluated on the LISA **test** folder. The original code pointed them at the training folder.
- Every ResNet18 run starts from fresh ImageNet weights. Previously, one model object was re-fine-tuned run after run.
- The LISA adversarial-training set is generated from LISA images.
- The CNN outputs logits (the duplicate Softmax is removed).
- The attack uses the same probability-space objective for both architectures.
- Adversarial images are clipped to the valid pixel range [0, 1].
- The ASR counts only non-target images, with no `abs()` approximation.
- Global seeding is added, and `torch.load(..., weights_only=False)` is used for PyTorch 2.6 and later.
- The shared LISA folder contained 10 duplicate uploads (`… (1).png`) that crashed label parsing; they are removed after loading.

## How to run

1. Open `Adversarial_Attacks_On_TSR_fixed.ipynb` in Google Colab with a **GPU runtime**.
2. Put the LISA data in `./data/LISA/{training,testing}`, with files named `<id>_<className>.png`, and place `mask_trial.png` in the working directory. GTSRB downloads automatically through torchvision.
3. Create `./model/`, `./data/adversarial_GTSRB/` and `./data/adversarial_LISA/`.
4. Run sections 1–3 to train and attack the baseline models.
5. Set `data_prep = True` to regenerate the adversarial-training images with the fixed code.
6. Run section 4 for adversarial training.

The notebook's first cells mount Google Drive, copy the data, and save models and `results.jsonl` to Drive so an interrupted run resumes where it stopped. On a free Colab T4 the full run takes roughly 8–10 GPU hours.

## Known limitations (still true after the fixes)

- The same mask and target label are used for adversarial training and for the attack. The defense is evaluated against the exact attack it was trained on, which overstates its robustness.
- Adversarial examples for training the **ResNet** models are generated against the **CNN** (a transfer setting). This is disclosed rather than changed.
- The attack uses only 10 optimization steps per image with no regularization (λ = 0), and each configuration is run with a single seed.

## References

1. K. Eykholt et al. *Robust Physical-World Attacks on Deep Learning Visual Classification.* CVPR, 2018.
2. C. Szegedy et al. *Intriguing Properties of Neural Networks.* ICLR, 2014.
3. I. Goodfellow, J. Shlens, C. Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR, 2015.
4. A. Kurakin, I. Goodfellow, S. Bengio. *Adversarial Examples in the Physical World.* arXiv:1607.02533, 2016.
5. T. Brown et al. *Adversarial Patch.* arXiv:1712.09665, 2017.
6. A. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR, 2018.
7. J. Stallkamp et al. *Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition.* Neural Networks, 2012 (GTSRB).
8. A. Møgelmose, M. Trivedi, T. Moeslund. *Vision-Based Traffic Sign Detection and Analysis for Intelligent Driver Assistance Systems.* IEEE T-ITS, 2012 (LISA).
