import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BeitImageProcessor, BeitForImageClassification
from tqdm.notebook import tqdm

def generate_predictions(test_images_dir, train_df, model_path, submission_file):
    """
    Generate predictions for test images using a trained BEiT model and save the predictions to a submission file.

    Parameters:
    test_images_dir (str): Directory path containing test images.
    train_df (pd.DataFrame): DataFrame containing training labels.
    model_path (str): Path to the trained BEiT model checkpoint.
    submission_file (str): Path to save the submission file.
    """
    # Custom sorting function to sort filenames numerically
    def numerical_sort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    # Create a DataFrame for the test images
    test_image_names = sorted([f.split('.')[0] for f in os.listdir(test_images_dir) if f.endswith('.jpg')], key=numerical_sort)
    test_df = pd.DataFrame({'image_name': test_image_names})

    # Custom dataset class for test images
    class TestImageDataset(Dataset):
        def __init__(self, df, img_dir, processor):
            self.df = df
            self.img_dir = img_dir
            self.processor = processor

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0] + '.jpg')
            image = Image.open(img_name).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            # Ensure the 'pixel_values' key matches expected input for BEiT model
            inputs['pixel_values'] = inputs.pop('pixel_values').squeeze()
            file_name = os.path.basename(img_name)  # Extract only the file name
            return inputs, file_name

    # Initialize the processor and load the model
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained(
        'microsoft/beit-base-patch16-224-pt22k-ft22k',
        num_labels=train_df.shape[1] - 1,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create test dataset and dataloader
    test_dataset = TestImageDataset(test_df, test_images_dir, processor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Generate predictions
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, batch_image_names = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    # Save the predictions to a submission file
    with open(submission_file, 'w') as file:
        for i, image_name in enumerate(test_image_names):
            line = image_name + ' ' + ' '.join(map(str, (np.array(predictions[i]) > 0.5).astype(int))) + '\n'
            file.write(line)

    print(f"Predictions saved to {submission_file}")

# Example usage
test_images_dir = 'SCSPAR24_Testdata/SCSPAR24_Testdata'
train_df = pd.read_csv("train_df.csv")
model_path = 'best_model_par.pth'
submission_file = 'submission.txt'

generate_predictions(test_images_dir, train_df, model_path, submission_file)
