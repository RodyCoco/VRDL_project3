# VRDL_project3

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
Please put images directory as following
```
${ROOT}
  +- dataset
  |  +- test
  |  +- train
  |  +- test_img_ids.json
```
## Hardware

RTX A5000

## Preprocessing
Before train the model, run this command to reorganize the directory of image:
```
python split_data.py
```
And you will get following directory
```
${ROOT}
  +- dataset
  |  +- test
  |  +- train
  |  +- train_images
  |  +- test_img_ids.json
```
## Training

To train the model, run this command:

```
python main.py
```

## Evaluation

To evaluate my model, run:

```eval
python evaluation.py
```
## Model Weight Link

