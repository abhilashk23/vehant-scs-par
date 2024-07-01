import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import numpy as np
import re

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

def test_model(test_img_dir, model, output_file):
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of test images
    image_files = sorted(os.listdir(test_img_dir), key=lambda x: int(re.search(r'\d+', os.path.splitext(x)[0]).group()) if re.search(r'\d+', os.path.splitext(x)[0]) else 9999999)

    results = []

    # Iterate over test images
    for img_name in image_files:
        img_path = os.path.join(test_img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension

            # Get model prediction
            model.eval()
            with torch.no_grad():
                image = image.to(next(model.parameters()).device)  # Move image to the same device as the model
                output = model(image)
                predicted = torch.sigmoid(output).round().cpu().numpy().astype(int).flatten()

            # Append result (image name without extension and one-hot encoded labels)
            results.append((os.path.splitext(img_name)[0], predicted))
        
        except UnidentifiedImageError:
            print(f"Skipping {img_path} as it is not a recognized image file.")
            continue

    # Write results to output file
    with open(output_file, 'w') as f:
        for i, (img_name, labels) in enumerate(results, 1):
            labels_str = ' '.join(map(str, labels))
            f.write(f"{i} {labels_str}\n")

# Example usage
def main():
    # Load the trained model
    backbone = models.resnet50(pretrained=True)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()  # Remove the final fully connected layer

    classifier = nn.Linear(num_features, 49)  # Assuming 49 classes
    model = FeatClassifier(backbone, classifier)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load('best_model_v2.pth', map_location=torch.device('cpu')))

    # Move model to device (GPU or CPU)
    device = torch.device("cpu")
    model = model.to(device)

    # Define test image directory and output file path
    test_img_dir = 'SCSPAR24_Testdata'
    output_file = 'submission.txt'

    # Generate predictions
    test_model(test_img_dir, model, output_file)

if __name__ == "__main__":
    main()
