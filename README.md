# Brain Tumour Classification Using CNN

## Setup

```shell
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip3 install -r requirements.txt
```

## Prediction

The trained model is located in `trained models`. To receive a prediction run:

```shell
python3 -m predict.py
```

Change `img_path` variable in `predict.py` to test other types of tumour. 

## Training

To train a new model, download the dataset from: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri and change `dataset_path` to the relative path. Then run:

```shell
python3 -m train.py
```

Uncomment lines 71-72 in `train.py` to produce evaluation metrics and graphs.

