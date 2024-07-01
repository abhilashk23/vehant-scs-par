import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tqdm
from transformers import SwinForImageClassification, AutoImageProcessor

def test_inference(img_dir, model_path, output_path):
    # Custom dataset class for test data
    class TestImageDataset(Dataset):
        def __init__(self, img_dir, transform=None, processor=None):
            self.img_dir = img_dir
            self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            self.transform = transform
            self.processor = processor

        def __len__(self):
            return len(self.img_names)

        def __getitem__(self, idx):
            img_name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file {img_path} not found.")
            image = Image.open(img_path).convert("RGB")
            
            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze()
            elif self.transform:
                image = self.transform(image)
            
            return image, img_name

    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')
    model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')

    # Modify the classifier to match the number of classes in the fine-tuned model
    num_classes = 49
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create dataset and data loader
    test_dataset = TestImageDataset(img_dir, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Set up device
    device = torch.device('cpu')
    model.to(device)

    results = []

    with torch.no_grad():
        for images, img_names in tqdm.tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.sigmoid(logits).cpu().numpy()
            one_hot_labels = (probs > 0.5).astype(int)

            for img_name, labels in zip(img_names, one_hot_labels):
                img_name = os.path.splitext(img_name)[0]  # Remove file extension
                result = [int(img_name)] + labels.tolist()  # Ensure img_name is an integer for sorting
                results.append(result)

    # Sort results by the first column
    results.sort(key=lambda x: x[0])

    # Save results to .txt file
    with open(output_path, 'w') as f:
        for result in results:
            f.write(' '.join(map(str, result)) + '\n')

    print(f"Inference results saved to {output_path}")

# Usage
img_dir = 'SCSPAR24_Testdata'
model_path = 'best_model_swin.pth'
output_path = 'submission.txt'

test_inference(img_dir, model_path, output_path)
