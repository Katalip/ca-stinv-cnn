## Setup
  
1. Create an environment:
```
conda create -n <env_name> python=3.7
```
2. This version of pytorch requires separate installation (creating env from env.yml file or requirements.txt does not find it)
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other necessary dependencies
```
pip install -r requirements.txt
```
