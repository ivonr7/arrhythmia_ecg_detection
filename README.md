# arrhythmia_ecg_detection
Classifying Arrythmia from 12 Lead ECG recordings


## Downloading Data

Our Dataset is [Arrythmi in Children and Adults]("https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/") from [PhysioNet]("https://physionet.org").  To download the dataset use the command
```bash
wget -r -N -c -np https://physionet.org/files/leipzig-heart-center-ecg/1.0.0/

```
this will save the dataset to the **physionet.org** directory in your current directory


## Setting up Project 
Create a virtual environment in python using the tool of your choice.  For example:
```bash
python -m venv venv
source venv/bin/activate # Will need to change if on windows!
```
Then install all dependencies listed in the [requirements]("requirements.txt") file.
```
pip install -r requirements.txt
```
## Place dataset files:
The necessary files to work with the dataset can be found [here](https://1sfu-my.sharepoint.com/personal/imv_sfu_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fimv%5Fsfu%5Fca%2FDocuments%2Ffall2025%2Fdataset%2Ezip&parent=%2Fpersonal%2Fimv%5Fsfu%5Fca%2FDocuments%2Ffall2025&ga=1).
- To regenerate them run
```
scripts/combine_dataset.py
```
- Pass the –help flag it will explain the parameters 
