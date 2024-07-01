
# Person Attribute Recognition

This project aims to perform Person Attribute Recognition (PAR) using a pretrained model and fine tunned on downstream task. Below are the steps to set up and run the project.

## Requirements

Before running the code, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Steps to Run the Project

### 1. Data Augmentation

Run the data augmentation script to preprocess and augment the training images:

train and val folder contain the image which will be augmented for further processing

```bash
python augmentation.py
```

### 2. Utility Functions

Execute the labels functions script which contains code to map label to augmented images and save in csv functions for the project:

```bash
python labels.py
```

### 3. Model Training

Train the model :


```bash
python train.py
```

### 4. Inference

Perform inference on the test data using the trained model:

```bash
python inference.py
```

## Summary

1. Augmentation of the training images was performed.
2. A pretrained model from Hugging Face was used.
3. The model was fine-tuned on the downstream task of Person Attribute Recognition (PAR).
4. The model was then tested on the provided test data.

## Files

- `requirements.txt`: Lists the required packages.
- `augmentation.py`: Script for data augmentation.
- `labels.py`: Contains helper functions.
- `train.py`: Script for training the model.
- `inference.py`: Script for performing inference on the test data.
- `submision.txt`: prediction on test data
- `best_model_swin.pth`: model trained and saved.

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```
## Disclaimer: Shared dataset annotation is the legal property of Vehant Technology; strictly for challenge use only – refrain from using it for any other application or publication, ensure data security, and delete after challenge completion. 
The dataset is Not shared Because of the above Disclaimer , still you can have look into code for reference.

## Contact

For any questions or issues, please contact 

Shubham kale 
shubham23094@iiitd.ac.in

Shashank Sharma 
shashank23088@iiitd.ac.in

Abhilash Khuntia
abhilash23007@iiitd.ac.in

---

This README provides an overview of the project structure and instructions for running each step. Make sure to follow the steps in order to ensure proper execution of the project.
