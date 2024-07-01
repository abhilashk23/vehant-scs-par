import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def augment_images(original_images_dir, augmented_images_dir, num_augmented_images=10):
    """
    Augment images in the specified directory and save them to another directory.

    Parameters:
    original_images_dir (str): Directory path where original images are located.
    augmented_images_dir (str): Directory path where augmented images will be saved.
    num_augmented_images (int): Number of augmented images to generate per original image.

    Returns:
    int: Total number of augmented images generated.
    """
    # Create the directory if it doesn't exist
    os.makedirs(augmented_images_dir, exist_ok=True)

    # Parameters for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # List all images in the original directory
    image_files = os.listdir(original_images_dir)

    # Initialize a counter for total augmented images generated
    total_augmented_images = 0

    # Iterate through each original image
    for image_file in image_files:
        # Load the image
        img = load_img(os.path.join(original_images_dir, image_file))
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, height, width, channels) for the datagen

        # Generate augmented images
        for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
            augmented_img_array = batch[0].astype('uint8')  # Convert to uint8 for saving as image
            
            # Generate the filename for the augmented image
            filename, ext = os.path.splitext(image_file)
            augmented_image_name = f"{filename}_augmented_{i+1}{ext}"  # Use original filename with "_augmented_" and index
            
            # Save augmented image
            augmented_img = array_to_img(augmented_img_array)
            augmented_img.save(os.path.join(augmented_images_dir, augmented_image_name))

            total_augmented_images += 1

            if i+1 >= num_augmented_images:
                break  # Exit the loop after generating the specified number of augmented images

    print(f"Generated {total_augmented_images} augmented images in {augmented_images_dir}")
    return total_augmented_images

# Example usage
original_images_dir = 'VRL_challenge_PAR/images'
augmented_images_dir = 'Pipeline/augmented_images'
num_augmented_images = 10

augment_images(original_images_dir, augmented_images_dir, num_augmented_images)