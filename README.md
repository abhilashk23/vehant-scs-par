
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

```bash
python augmentation.py
```

### 2. Utility Functions

Execute the utility functions script which contains helper functions for the project:

```bash
python utils.py
```

### 3. Model Training

Train the model using the pretrained Beit model:

```python
from transformers import BeitImageProcessor, BeitForImageClassification

processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k-ft22k',
    num_labels=train_df.shape[1] - 1,
    ignore_mismatched_sizes=True
)
# Additional training code here
```

Run the model training script:

```bash
python model.py
```

### 4. Inference

Perform inference on the test data using the trained model:

```bash
python inference.py
```

## Summary

1. Augmentation of the training images was performed.
2. A pretrained Beit model from Hugging Face (`microsoft/beit-base-patch16-224-pt22k-ft22k`) was used.
3. The model was fine-tuned on the downstream task of Person Attribute Recognition (PAR).
4. The model was then tested on the provided test data.
5. Pipeline folder will store the augmented data .

## Files

- `requirements.txt`: Lists the required packages.
- `augmentation.py`: Script for data augmentation.
- `utils.py`: Contains utility functions.
- `model.py`: Script for training the model.
- `inference.py`: Script for performing inference on the test data.
- `submision.txt`: submision on test data
- `best_model_par.pth`: model trained on saved.

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
