import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from tqdm import tqdm

# Load the dataset
csv_path = 'Pipeline/augmented_df.csv'
df = pd.read_csv(csv_path)

# Split the dataset into training, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        labels = self.df.iloc[idx, 1:].values.astype('float32')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels)
        # Ensure the 'pixel_values' key matches expected input for Beit model
        inputs['pixel_values'] = inputs.pop('pixel_values').squeeze()
        return inputs

# Initialize the processor and model
processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained(
    'microsoft/beit-base-patch16-224-pt22k-ft22k', 
    num_labels=train_df.shape[1] - 1,
    ignore_mismatched_sizes=True
)

# Create datasets
train_dataset = ImageDataset(train_df, 'Pipeline/augmented_images', processor)
val_dataset = ImageDataset(val_df, 'Pipeline/augmented_images', processor)
test_dataset = ImageDataset(test_df, 'Pipeline/augmented_images', processor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Custom callback for label-based mean accuracy (mA)
class LabelMeanAccuracy:
    def on_epoch_end(self, model, val_loader, device):
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                y_true.append(labels.cpu().numpy())
                y_pred.append(logits.cpu().numpy())

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)

        accuracies = []
        for i in range(y_true.shape[1]):
            accuracies.append(np.mean(y_true[:, i] == (y_pred[:, i] > 0.5)))
        mA = np.mean(accuracies)

        print(f' - val_label_mean_accuracy: {mA:.4f}')

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

# Training function
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training loop
num_epochs = 10
best_val_loss = float('inf')
label_mean_accuracy = LabelMeanAccuracy()

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    
    # Check if this is the best model so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_par.pth')
    
    # Custom callback to calculate label mean accuracy
    label_mean_accuracy.on_epoch_end(model, val_loader, device)

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model_par.pth'))

# Evaluate the model on the test set
model.eval()
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        y_true_test.append(labels.cpu().numpy())
        y_pred_test.append(logits.cpu().numpy())

y_true_test = np.vstack(y_true_test)
y_pred_test = np.vstack(y_pred_test)

# Calculate label-based mean accuracy (mA) for the test set
test_accuracies = []
for i in range(y_true_test.shape[1]):
    test_accuracies.append(np.mean(y_true_test[:, i] == (y_pred_test[:, i] > 0.5)))
test_label_mean_accuracy = np.mean(test_accuracies)

print(f'Test label-based mean accuracy (mA): {test_label_mean_accuracy:.4f}')
