import os
import pandas as pd

def load_process_and_merge(data_path, column_names, image_folder_path, output_csv_path):
    """
    Load the data, create a DataFrame with image filenames, merge the DataFrames,
    and save the merged DataFrame to a CSV file.

    Parameters:
    data_path (str): Path to the data file.
    column_names (list): List of column names for the DataFrame.
    image_folder_path (str): Path to the folder containing the image files.
    output_csv_path (str): Path to save the merged DataFrame CSV file.
    """
    # Load the data into a DataFrame
    data = pd.read_csv(data_path, sep=' ', header=None)

    # Extract image names and labels
    image_names = data.iloc[:, 0]
    labels = data.iloc[:, 1:]

    # Rename the columns
    labels.columns = column_names

    # Combine the image names and labels into a single DataFrame
    df = pd.concat([image_names, labels], axis=1)

    # Rename the first column to 'image_name'
    df.rename(columns={df.columns[0]: 'image_name'}, inplace=True)

    # List all image filenames in the folder
    image_files = os.listdir(image_folder_path)

    # Extract numeric ID and sort filenames based on this ID
    sorted_image_files = sorted(image_files, key=lambda x: int(x.split('_')[0]))

    # Create a new DataFrame with an image_name column
    image_df = pd.DataFrame(sorted_image_files, columns=['image_name'])

    # Extract the numeric ID from the image_name in image_df
    image_df['id'] = image_df['image_name'].apply(lambda x: int(x.split('_')[0]))

    # Merge image_df with train_df based on the extracted ID
    augmented_df = pd.merge(image_df, df, left_on='id', right_on='image_name', how='left')

    # Drop the redundant 'image_name_y' and 'id' columns
    augmented_df = augmented_df.drop(columns=['id', 'image_name_y'])

    # Rename columns for clarity
    augmented_df = augmented_df.rename(columns={'image_name_x': 'image_name'})

    # Save the merged dataframe to a new CSV file
    augmented_df.to_csv(output_csv_path, index=False)

    print(f"Saved merged DataFrame to {output_csv_path}")

# Example usage
# Define the column names
column_names = [
    'UBCOLOR_black', 'UBCOLOR_blue', 'UBCOLOR_green', 'UBCOLOR_orange', 'UBCOLOR_red', 
    'UBCOLOR_white', 'UBCOLOR_yellow', 'UBCOLOR_mix', 'UBCOLOR_other',
    'LBCOLOR_black', 'LBCOLOR_blue', 'LBCOLOR_green', 'LBCOLOR_red', 'LBCOLOR_yellow',
    'LBCOLOR_white', 'LBCOLOR_orange', 'LBCOLOR_other', 'LBCOLOR_mix',
    'UB_jacket', 'UB_kurta', 'UB_other', 'UB_Saree', 'UB_Shirt', 'UB_Suitwomen', 
    'UB_tshirt', 'UB_sweater',
    'LB_Leggings', 'LB_Salwar', 'LB_Shorts', 'LB_Trousers', 'LB_Jeans', 'LB_Saree', 
    'LB_other',
    'SLEEVES_long', 'SLEEVES_short', 'SLEEVES_none',
    'Carry_handbag', 'Carry_backpack', 'Carry_other',
    'Acc_headgear',
    'Foot_Sandals', 'Foot_shoes', 'Foot_slippers',
    'POSE_sitting', 'POSE_lying', 'POSE_standing',
    'VIEW_back', 'VIEW_front', 'VIEW_side'
]

# Path to the image folder
image_folder_path = 'Pipeline/augmented_images'

# Load the train.txt file into a DataFrame
data_path = 'VRL_challenge_PAR/train.txt'

# Output CSV path
output_csv_path = 'Pipeline/augmented_df.csv'

# Call the function to load, process, merge, and save the data
load_process_and_merge(data_path, column_names, image_folder_path, output_csv_path)
