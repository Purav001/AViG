import os
import shutil
import random

def create_dataset_subset(source_dir, dest_dir, train_images_per_class, val_images_per_class):
    if os.path.exists(dest_dir):
        print(f"Destination '{dest_dir}' already exists. Removing it for a fresh start.")
        shutil.rmtree(dest_dir)
    
    print(f"Creating subset in '{dest_dir}'...")
    
    for split, num_images in [('train', train_images_per_class), ('val', val_images_per_class)]:
        source_split_dir = os.path.join(source_dir, split)
        dest_split_dir = os.path.join(dest_dir, split)
        
        if not os.path.exists(source_split_dir):
            continue
            
        print(f"  Processing '{split}' split with {num_images} images per class...")
        class_names = os.listdir(source_split_dir)
        
        for class_name in class_names:
            source_class_dir = os.path.join(source_split_dir, class_name)
            dest_class_dir = os.path.join(dest_split_dir, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            
            all_images = os.listdir(source_class_dir)
            sampled_images = random.sample(all_images, min(len(all_images), num_images))
            
            for image_name in sampled_images:
                shutil.copyfile(os.path.join(source_class_dir, image_name), os.path.join(dest_class_dir, image_name))
                
    print("\nSubset dataset created successfully!")

if __name__ == '__main__':
    # --- CONTROL HOW MANY IMAGES YOU WANT HERE ---
    create_dataset_subset(
    source_dir='./cifar10_dataset',
    dest_dir='./cifar10_subset_medium',     # <-- New destination folder
    train_images_per_class=1000,          # <-- Increase to 1000
    val_images_per_class=100              # <-- Increase to 100
    )