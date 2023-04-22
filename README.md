## 🔬 Improving Stain Invariance of CNNs (Implementation)

> Full Title: Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training <br>
> Paper: To add link <br>

Thank you for expressing interest in our work

## 📖 Navigation
<details>
  <summary>Click me</summary>
  ## Setup

1. Create an environment:
```
conda create -n <env_name> python=3.9
```
2. This version of pytorch requires separate installation (creating env from env.yml file or requirements.txt does not find it)
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other necessary dependencies
```
pip install -r requirements.txt
```

</details>

- [Environment Setup](./docs/environment.md)
- [Repository Structure](./docs/structure.md)
- [Config File Explanation](./docs/config_explanation.md)
- [Datasets](./docs/datasets.md)
- [Train](./docs/train.md)
- [Evaluate](./docs/eval.md)
- [How to use the method with your/custom model](./docs/integration.md)