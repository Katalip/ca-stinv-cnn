
## Datasets
We provide the links below and give a short description of their origin.

**HPA + HuBMAP 2022**. Human Protein Atlas (HPA) is a Swedish-based program (make a link), and The Human BioMolecular Atlas Program (HuBMAP) details its data contributors (US) [here.](https://hubmapconsortium.org/hubmap-data/#:~:text=HuBMAP%20data%20was%20generated%20using,assay%20types%20used%20in%20each) [Description of the dataset.](https://www.biorxiv.org/content/10.1101/2023.01.05.522764v1) [Download link](https://zenodo.org/record/7545745#.Y-M5SXZBwal) It is important to mention that the test set was not available during this study and this download page has been created recently <br>

**The Nephrotic Syndrome Study Network (NEPTUNE)** is a North American multi-center consortium. We use a subset of this dataset that contains only glomeruli with annotations of Bowman’s space to match the training data. Samples were collected across 29 enrollment centers (US and Canada). 
[Description.](https://www.sciencedirect.com/science/article/pii/S0085253820309625)
The [download link](https://github.com/ccipd/DL-kidneyhistologicprimitives) is available at the bottom as online supplemental material (we use files named with 'glom_capsule').
 
**Academia and Industry Collaboration for Digital Pathology (AIDPATH)** is a Europen project. The data is collected in Spain and hosted by Mendeley. [Description.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7058889/#fn2) [Download](https://data.mendeley.com/datasets/k7nvtgn2x6/3)

WSIs in **HuBMAP21 Kidney** should come from the data contributors that can be viewed by the link provided above. [Description.](https://www.biorxiv.org/content/10.1101/2021.11.09.467810v1) [Download](https://github.com/cns-iu/ccf-research-kaggle-2021) (Data section)

We do not perform any specific preprocessing. Training images are resized to 768x768, while test samples are resized to sizes that match stats (pixel size, magnification) of the train data.
NEPTUNE images to 480x480
AIDPATH samples to 256x256
HuBMAP21 Kidney WSIs to 224x224
