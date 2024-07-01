import os
import csv

def create_augmented_image_csv(augmented_folder, label_file, output_csv):
    # Step 1: Read the label file into a dictionary
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]
            label = parts[1:]  # Remaining parts are the labels
            labels[image_name] = label

    # Step 2: Walk through the augmented images folder to get image names
    augmented_images = []
    for root, dirs, files in os.walk(augmented_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Add other image extensions if needed
                augmented_images.append(file)

    # Step 3: For each augmented image, extract the base name and find its label
    data = []
    for image in augmented_images:
        base_name = '_'.join(image.split('_')[:-2])  # Extract base name before '_augmented_X'
        if base_name in labels:
            data.append([image] + labels[base_name])
        else:
            data.append([image] + ['']*49)  # If no label found, leave it empty or handle as needed

    # Step 4: Write the data to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['image_name'] + [f'Label{i}' for i in range(1, 50)]
        writer.writerow(header)
        writer.writerows(data)

    print(f"CSV file created successfully at {output_csv}")

# Example usage:
if __name__ == "__main__":
    augmented_folder = 'augmented_image_train'  # Update with your actual path
    label_file = 'train.txt'              # Update with your actual path
    output_csv = 'augmented_image_train.csv'    # Update with your desired output path
    augmented_folder_val = 'augmented_image_val'  # Update with your actual path
    label_file = 'train.txt'              # Update with your actual path
    output_csv_val = 'augmented_image_val.csv'

    create_augmented_image_csv(augmented_folder, label_file, output_csv)
    create_augmented_image_csv(augmented_folder_val, label_file, output_csv_val)
