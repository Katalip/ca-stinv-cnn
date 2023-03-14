## Improving Stain Invariance of CNNs (Implementation)

> Full title: Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training <br>
> Paper: To do (under review) <br> 
> Slide: To do <br>

> **Abstract**:
> *Variability in staining protocols, which can result in a diverse set of whole slide images (WSI) with different slide preparation techniques, chemicals, and scanner configurations, presents a significant challenge for adapting convolutional neural networks (CNNs) for computational pathology applications. This distribution shift can negatively impact the performance of deep learning models on unseen samples, especially in the task of semantic segmentation. In this study, we propose a method for improving the generalizability of CNNs to stain changes in a single-source setting. Our approach uses a channel attention mechanism that detects stain-specific features and a modified stain-invariant training scheme based on recent findings. We evaluate our method on multi-center, multi-stain datasets and demonstrate its effectiveness through interpretability analysis. Our approach achieves substantial improvements over baselines and competitive performance compared to other methods, as measured by various evaluation metrics. We also show that combining our method with stain augmentation leads to mutually beneficial results and outperforms other techniques. Overall, our study makes several important contributions to the field of computational pathology, including the proposal of a novel method for improving the generalizability of CNNs to stain changes in semantic segmentation and the modification of the previously proposed stain-invariant training scheme.* <br>

Thank you for expressing interest in our work.

## 📖 Navigation
- [Environment Setup](./docs/environment.md)
- [Repository Structure](./docs/structure.md)
- [Config File Explanation](./docs/config_explanation.md)
- [Datasets](./docs/datasets.md)

#### To do:
- [ ] Add other configs
- [ ] Add other datasets in testing script
- [ ] Add training logs
- [x] Add dataset links
- [x] Add config explanation
- [ ] Check environment setup on other OS 
- [ ] Add instruction on how to adapt the method for your own model
- [ ] Check macenko_torch.py with newer pytorch versions
- [ ] Check whether wandb login info gets stored or not
- [ ] Automate model selection
- [ ] List potential small erros due to hyperparameter changes
- [ ] Add test time stain normalization in val.py
- [ ] Add ISW license

