# scTRaCT v0.1.0

**scTRaCT** is a supervised transformer-based deep learning framework that integrates log-normalized gene expression with complementary distance-based features derived from Multiple Correspondence Analysis (MCA). By transforming continuous expression data into a metric that quantifies the association between genes and cells, scTRaCT enriches the input representation, capturing subtle transcriptional differences that are critical for distinguishing closely related cell types. Processed through a Transformer-based architecture with self-attention mechanisms, our model effectively learns long-range dependencies and complex gene interactions

![](Images/scTRaCT_overview.png)



## Installation
you caninstall the scTRaCT package following next steps: 

```
pip install git+https://github.com/msmalmir/scTRaCT.git
```
OR

```
git clone https://github.com/msmalmir/scTRaCT.git
cd scTRaCT
pip install 
```

## Usage
After installing the package you need to prepare your dataset to feed into the model. As scTRaCT is working only with HVGs please use the preprocessed.py to filter your dataset 


<!-- 
## Installation

To install the latest version directly from GitHub, use the following command:

```
pip install git+https://github.com/msmalmir/scTransID.git
```

## Data Input
**scTransID** will require you to give two directory where you saved the refrence and quety datasets. Both of them should be adata files and please be sure the refrence data includes 'celltype' information for each cell. 

## Output 
Output will be the predicted celltypes for the samples in query dataset. 

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- scanpy

Install these dependencies via `pip`:

```
pip install torch scikit-learn scanpy
```

## Tutorial
For a complete usage example pleae refer to **Tutorial** folder in this GitHub repository, you can directly access it from [here.](https://github.com/msmalmir/scTransID/tree/main/Tutorial) -->
