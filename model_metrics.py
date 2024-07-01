import os
import numpy as np
import time
import psutil
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BeitImageProcessor, BeitForImageClassification

def train_beit_model(img_dir_train, csv_path_train, img_dir_val, csv_path_val, num_epochs=15, batch_size=8, lr=1e-5):
    
    # Load the dataset
    train_df = pd.read_csv(csv_path_train)
    val_df = pd.read_csv(csv_path_val)

    # Custom dataset class
    class ImageDataset(Dataset):
        def __init__(self, df, img_dir, transform=None, processor=None):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform
            self.processor = processor

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
            if not os.path.exists(img_name):
                raise FileNotFoundError(f"Image file {img_name} not found.")
            image = Image.open(img_name).convert("RGB")
            
            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze()
            elif self.transform:
                image = self.transform(image)
            
            labels = self.df.iloc[idx, 1:].values.astype('float32')
            return image, torch.tensor(labels)

    # Initialize the Beit processor and model
    num_labels = train_df.shape[1] - 1
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained(
        'microsoft/beit-base-patch16-224-pt22k-ft22k', 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Create datasets
    train_dataset = ImageDataset(train_df, img_dir_train, processor=processor)
    val_dataset = ImageDataset(val_df, img_dir_val, processor=processor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Custom callback for label-based mean accuracy (mA)
    class LabelMeanAccuracy:
        def on_epoch_end(self, model, val_loader, device):
            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for batch in tqdm(val_loader):
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training function
    def train_one_epoch(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits if hasattr(outputs, 'logits') else outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs.logits if hasattr(outputs, 'logits') else outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.numel()
        
        accuracy = correct_predictions / total_predictions
        return total_loss / len(train_loader), accuracy

    # Validation function
    def validate(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.logits if hasattr(outputs, 'logits') else outputs, labels)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs.logits if hasattr(outputs, 'logits') else outputs) > 0.5
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.numel()
        
        accuracy = correct_predictions / total_predictions
        return total_loss / len(val_loader), accuracy

    # Training loop
    label_mean_accuracy = LabelMeanAccuracy()
    best_val_loss = float('inf')

    # Resource utilization logging
    log_data = []

    for epoch in range(num_epochs):
        start_time = time.time()
        cpu_usage_start = psutil.cpu_percent(interval=None)
        memory_usage_start = psutil.virtual_memory().percent
        
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        cpu_usage_end = psutil.cpu_percent(interval=None)
        memory_usage_end = psutil.virtual_memory().percent
        epoch_duration = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_swin.pth')
        
        # Custom callback to calculate label mean accuracy
        label_mean_accuracy.on_epoch_end(model, val_loader, device)

        # Log epoch details
        log_data.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'epoch_duration': epoch_duration,
            'cpu_usage_start': cpu_usage_start,
            'cpu_usage_end': cpu_usage_end,
            'memory_usage_start': memory_usage_start,
            'memory_usage_end': memory_usage_end
        })

    # Save log data to CSV
    log_df = pd.DataFrame(log_data)
    log_df.to_csv('training_log.csv', index=False)

    # Plot and save learning curves and accuracies
    def plot_metrics(log_df, save_dir):
        epochs = log_df['epoch']

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, log_df['train_loss'], label='Training Loss')
        plt.plot(epochs, log_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, log_df['train_accuracy'], label='Training Accuracy')
        plt.plot(epochs, log_df['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(os.path.join(save_dir, 'metrics.png'))
        plt.close()

    save_dir = 'visualizations'
    os.makedirs(save_dir, exist_ok=True)
    plot_metrics(log_df, save_dir)

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model_swin.pth'))
    print("Finished training Swin Transformer model.")

# Usage
img_dir_train = 'augmented_image_train'
csv_path_train = 'augmented_image_train.csv'
img_dir_val = 'augmented_image_val'
csv_path_val = 'augmented_image_val.csv'

train_beit_model(img_dir_train, csv_path_train, img_dir_val, csv_path_val)