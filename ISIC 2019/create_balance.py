# Path: create_balanced_subset.py (Corrected Version)
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

def create_balanced_subset(source_dir, csv_path, dest_dir, classes_to_keep, balance_target_class, balance_count, val_split_size=0.2):
    """
    Creates a balanced, split (train/val) dataset subset from the ISIC 2019 data.
    """
    print(f"Creating balanced subset in '{dest_dir}'...")

    if os.path.exists(dest_dir):
        print(f"Destination '{dest_dir}' already exists. Removing it for a fresh start.")
        shutil.rmtree(dest_dir)

    # 1. Load the ground truth CSV
    df = pd.read_csv(csv_path)
    
    # --- START OF THE FIX ---
    # First, determine the true class for ALL images from the one-hot columns
    # Get all class columns by assuming 'image' is the only non-class column
    all_class_columns = [col for col in df.columns if col != 'image']
    df['class'] = df[all_class_columns].idxmax(axis=1)

    # Now, filter the DataFrame to only include the rows with our desired classes
    df = df[df['class'].isin(classes_to_keep)]
    # --- END OF THE FIX ---
    
    print(f"Found {len(df)} total images for the specified classes.")

    # 2. Balance the dataset by downsampling the target class
    df_others = df[df['class'] != balance_target_class]
    df_target = df[df['class'] == balance_target_class]

    print(f"Downsampling '{balance_target_class}' from {len(df_target)} to {balance_count} images.")
    df_target_sampled = df_target.sample(n=balance_count, random_state=42)

    df_balanced = pd.concat([df_others, df_target_sampled])
    print(f"Total images in balanced set: {len(df_balanced)}")
    print("\nFinal class distribution in the subset:")
    print(df_balanced['class'].value_counts())

    # 3. Create stratified train/validation splits
    images = df_balanced['image'].values
    labels = df_balanced['class'].values
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=val_split_size, random_state=42, stratify=labels
    )
    
    print(f"\nSplitting into {len(train_images)} training and {len(val_images)} validation images.")

    # 4. Copy files into the correct train/val folder structure
    for split, image_list, label_list in [('train', train_images, train_labels), ('val', val_images, val_labels)]:
        split_dir = os.path.join(dest_dir, split)
        print(f"\nProcessing '{split}' split...")
        for image_name, label in tqdm(zip(image_list, label_list), total=len(image_list)):
            class_dir = os.path.join(split_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            
            source_path = os.path.join(source_dir, f"{image_name}.jpg")
            dest_path = os.path.join(class_dir, f"{image_name}.jpg")
            
            if os.path.exists(source_path):
                shutil.copyfile(source_path, dest_path)
            else:
                print(f"Warning: Source image not found at {source_path}")

    print("\nBalanced subset dataset created successfully!")


if __name__ == '__main__':
    SOURCE_IMAGE_DIR = './ISIC_2019_Training_Input' 
    CSV_FILE_PATH = './ISIC_2019_Training_GroundTruth.csv'
    DESTINATION_DIR = './isic_balanced_best'
    CLASSES = ['MEL', 'BCC', 'NV', 'BKL']
    MAJORITY_CLASS = 'NV'
    BALANCE_TARGET_COUNT = 4522
    
    create_balanced_subset(
        source_dir=SOURCE_IMAGE_DIR,
        csv_path=CSV_FILE_PATH,
        dest_dir=DESTINATION_DIR,
        classes_to_keep=CLASSES,
        balance_target_class=MAJORITY_CLASS,
        balance_count=BALANCE_TARGET_COUNT
    )