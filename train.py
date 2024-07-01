import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import os

# Define the ScaledBCELoss class
class ScaledBCELoss(nn.Module):
    def __init__(self, sample_weight=None, size_sum=True, scale=30, tb_writer=None):
        super(ScaledBCELoss, self).__init__()
        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.pos_scale = scale
        self.neg_scale = scale
        self.tb_writer = tb_writer

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        logits = logits * targets * self.pos_scale + logits * (1 - targets) * self.neg_scale

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))

        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)
            loss_m = (loss_m * sample_weight.cuda())

        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        return [loss], [loss_m]

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        labels = np.array(self.annotations.iloc[idx, 1:], dtype=np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(labels)

# Define the FeatClassifier
class FeatClassifier(nn.Module):
    def __init__(self, backbone, classifier, dropout=0.5):
        super(FeatClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.classifier(features)
        return out

# Main function to train and evaluate the model
def train_and_evaluate(train_csv, train_img_dir, val_csv, val_img_dir, num_epochs=10, batch_size=16, learning_rate=0.001, scale=30, model_save_path="best_model.pth"):
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom DataLoader
    train_data = CustomDataset(csv_file=train_csv, root_dir=train_img_dir, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    val_data = CustomDataset(csv_file=val_csv, root_dir=val_img_dir, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the pre-trained model
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()  # Remove the final fully connected layer

    classifier = nn.Linear(num_features, 49)  # Assuming 49 classes
    model = FeatClassifier(backbone, classifier)
    model = model.cuda()

    # Load pre-trained weights
    checkpoint = torch.load('best_model.pth')
    if 'state_dicts' in checkpoint:
        state_dict = checkpoint['state_dicts']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()

    # Define loss function (using ScaledBCELoss) and optimizer
    criterion = ScaledBCELoss(scale=scale)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training function
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
        best_val_accuracy = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, _ = criterion(outputs, labels)
                loss[0].backward()
                optimizer.step()
                running_loss += loss[0].item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_mean_accuracy = evaluate_model(model, train_loader)
            val_mean_accuracy = evaluate_model(model, val_loader)
            print(f'Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Train Mean Accuracy: {train_mean_accuracy:.4f} - Val Mean Accuracy: {val_mean_accuracy:.4f}')
            
            # Save the model if validation accuracy improves
            if val_mean_accuracy > best_val_accuracy:
                best_val_accuracy = val_mean_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at epoch {epoch} with validation accuracy: {val_mean_accuracy:.4f}")

            scheduler.step()

    # Evaluation function
    def evaluate_model(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                predicted = torch.sigmoid(outputs).round()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / (total * labels.size(1))

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Example usage
train_and_evaluate(
    train_csv='augmented_image_train.csv',
    train_img_dir='augmented_image_train',
    val_csv='augmented_image_val.csv',
    val_img_dir='augmented_image_val',
    num_epochs=30,
    batch_size=16,
    learning_rate=0.001,
    scale=30,
    model_save_path='best_model_v2.pth'
)
