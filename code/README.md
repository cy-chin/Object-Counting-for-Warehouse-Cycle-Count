# Setting up the data/ folder before Jupyter notebook execution

> Authors: Chung Yau

---

## Background
The data/ folder needs to contain 2 files downloaded from Amazon Bin Image Challenge. 
1. metadata in sqlite3 - 800+ MByte
2. bin images in jpeg - 3.37+ GByte

These data is not committed to github due to the large file size. However, they need to be presence in the data/ folder for the scripts on jupyter notebook to be executable.  

## Manual setup to-do
1. download the data from [Kaggle](https://www.kaggle.com/datasets/williamhyun/amazon-bin-image-dataset-536434-images-224x224/data) 
2. unzip
3. Place the metadata to `data/` folder
4. Place the 500K+ bin images to `data/bin-images-224x224/` folder

## Download direct from Kaggle
1. Dowonload direct from Kaggle API, with command 
`! kaggle datasets download williamhyun/amazon-bin-image-dataset-536434-images-224x224`
2. After download, unzip into `data/bin-images-224x224` folder

## Data with large file size
1. The require dataset to run `02_ModelEval_Baseline_v1.ipynb` and `03_ModelParameterTuning_VGG.ipynb` can be downloaded from Google Drive https://drive.google.com/drive/folders/11ozp-Ra8zHGhAxDDMCWJODr2srJzOSbu?usp=sharing