## 📖 Introduction

Adversarial attacks pose a severe security threat to advanced visual object trackers based on Deep Neural Networks (DNNs). While existing methods predominantly focus on adding imperceptible perturbations to image pixels, they overlook a highly accessible attack surface: the **initial annotated bounding box**.

This repository introduces **BboxAtt**, a novel black-box adversarial annotation generation method. Instead of manipulating image content, BboxAtt deliberately injects constrained perturbations solely into the coordinates of the initial bounding box in the first frame. Based on a particle filter algorithm, this method optimizes the bounding box to induce severe tracking failures in subsequent frames while ensuring the adversarial annotation remains visually stealthy and plausible.


## ✨ Key Contributions

* **New Perspective:** Shifts the adversarial attack paradigm in VOT from pixel-level tampering to perturbing the initial bounding box, exploiting the inherent ambiguity in manual annotations.
* **Effective Black-Box Method:** Proposes a particle filter-based algorithm (incorporating fitness evaluation, resampling, and mutation) to search for the optimal adversarial bounding box without requiring access to the victim tracker's internal parameters.
* **Broad Generalizability:** Demonstrates significant tracking degradation across diverse architectures (e.g., CNN-based, Transformer-based, spatio-temporal) and multi-modal tracking scenarios (RGB-E, RGB-T, RGB-D).

## 🚀 Supported Trackers & Benchmarks

Our evaluation encompasses a diverse set of 9 state-of-the-art victim models across 7 benchmarks:

* **RGB-based Trackers:** SiamRPN++, SiamBAN, STARK, **OSTrack**, ODTrack, HIPTrack
* **Multi-modal Trackers:** ViPT, SDSTrack, UnTrack
* **Benchmarks:** OTB100, LaSOT, GOT-10k, TrackingNet, VisEvent, LasHeR, DepthTrack

## 🛠️ Environment Setup

The codebase is built on **PyTorch** and optimized for seamless deployment. 

## Evaluation
```bash
python -u SiameseRPN++/tools/test.py
