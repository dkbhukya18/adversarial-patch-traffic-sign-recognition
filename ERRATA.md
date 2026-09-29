# Errata: original CIS 582 report and notebook

These issues were found by auditing `Adversarial_Attacks_On_TSR.ipynb` (last modified Apr 16, 2024) and the final report (Apr 2024). "As written" means the bug is in the shared notebook's code. The notebook doesn't contain saved outputs for those cells, so it's impossible to confirm exactly which version produced each reported number.

## Code issues (fixed in `Adversarial_Attacks_On_TSR_fixed.ipynb`)

| # | Issue | Effect on results |
|---|---|---|
| 1 | `lisa_aug_test_data` loads `./data/LISA/training` | Aug-LISA "test" accuracy (99.4% and 100%) and the attack numbers were likely measured on training images. **Invalid until re-run.** |
| 2 | A single `resnet18_model` object is reused for all 8 ResNet runs (`.fc` swapped each time) | Each "FT" model starts from the previous run's fine-tuned weights, e.g. LISA-ResNet starts from GTSRB weights. All ResNet rows are affected. |
| 3 | The LISA adversarial-data cell uses `DataLoader(gtsrb_training_data)` | As written, the LISA adversarial set would contain German signs with GTSRB labels. The saved `adversarial_LISA` files in Drive have labels up to 43 and indices around 5,000, which match LISA, so the saved data appears correct. The code must still be fixed for reproducibility. |
| 4 | The CNN ends in `nn.Softmax()` and is trained with `CrossEntropyLoss` | Softmax is applied twice, so the loss plateaus around 2.80. The models still reach about 96–98% accuracy, but gradients for the attack are distorted. |
| 5 | The attack takes the MSE between the model output and a one-hot vector | For the CNN that output is probabilities; for ResNet it is raw logits. The two architectures therefore faced different attack objectives, which confounds the CNN-vs-ResNet comparison. |
| 6 | The perturbed image is never clipped to [0, 1] | The perturbations can't be printed as-is, which contradicts the "physical sticker" claim. |
| 7 | ASR is computed as `abs(match_target − ignore_target)/(size − ignore_target)` | This approximation can miscount. It is replaced by (non-target images predicted as the target) / (non-target images). |
| 8 | No random seeds are set | Results can't be reproduced. |
| 9 | The GTSRB adversarial preview uses the LISA label map | Only the preview titles are wrong. |
| 10 | The shared `data/LISA/training` folder contains duplicate uploads (e.g. `…_turnRight (1).png`) | The label parser crashes on them (`KeyError: 'turnRight (1)'`); the folder has 5,509 files instead of 5,499. The fixed notebook deletes these copies after loading the data. |

## Report text issues

1. **The dataset size is wrong.** The torchvision GTSRB `train` split has **26,640** images (confirmed in the notebook log), not 39,209, which is the count for a different distribution of the dataset. The LISA counts (5,499 train, 2,356 test) are correct once the 10 duplicate files are removed.
2. **Augmentation is described incorrectly.** The code rotates up to **5°** (the report says 10°), and its color jitter also changes **saturation and hue**.
3. **Figure 8 can't be reproduced.** It shows about 1,500 images per class after augmentation, but on-the-fly torchvision transforms don't change class counts, and the notebook has no rebalancing step. Remove the figure or produce it with code.
4. **Figure 6 is not ResNet18.** It shows attention modules and a distillation path from a different architecture.
5. **The method is overstated as "RP2" with physical perturbations.** The implementation has no expectation over transformations and no printed tests. Describe it as a digital, RP2-style masked attack.
6. **Regularization is described but not used.** Section 3 describes L1/L2 regularization, but every experiment ran with λ = 0.
7. **The lowest clean accuracy is misquoted.** The text says 95.9% for ResNet18 on GTSRB, but Table 1 shows **95.6%** without adversarial training.
8. **"Maximum improvement for GTSRB-CNN" contradicts Table 1.** The table shows larger gains for Aug-LISA-CNN (+55.7 points) and Aug-LISA-ResNet (+47.7) than for GTSRB-CNN (+45.5). The LISA-aug numbers are suspect anyway (code item 1).
9. **The adversarial training setup is only partly disclosed.** The report notes that the same mask was used for training and attack (good), but it doesn't say that the ResNet models were adversarially trained on examples generated against the CNN, or that the target label was fixed at 2.
10. **The intro overclaims.** It promises a "Trustworthy AI framework", but no framework is developed.
11. **Formatting.** There are two sections numbered 4.3, and Table 1's column grouping (with/without adversarial training) is hard to read.

## Effect of the re-run

The corrected re-run (see README) replaces every number in the original Table 1. Notable changes: the Aug-LISA models score 98.3% (CNN) and 97.3% (ResNet18) on the real test set, not 99.4% and 100%; and adversarial training *reduced* robustness for both LISA CNNs rather than improving all models.
