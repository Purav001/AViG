# Path: prepare_fake_dataset.py
import os
from PIL import Image
import numpy as np

def create_fake_dataset(root_dir='fake_dataset', num_classes=1000, images_per_class_train=10, images_per_class_val=2):
    """Creates a fake dataset in the ImageFolder structure with random noise images."""
    
    print(f"Creating fake dataset in '{root_dir}'...")
    
    for split in ['train', 'val']:
        split_path = os.path.join(root_dir, split)
        images_per_class = images_per_class_train if split == 'train' else images_per_class_val
        
        print(f"  Generating '{split}' split with {images_per_class} images per class...")
        
        for i in range(num_classes):
            class_name = f'n{i:08d}'  # Mimic ImageNet class folder names
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            for j in range(images_per_class):
                # Create a 224x224 random noise image
                random_image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(random_image_array)
                
                image_filename = f'fake_{i}_{j}.jpeg'
                image_path = os.path.join(class_path, image_filename)
                image.save(image_path)
        
        print(f"  Finished '{split}' split.")
        
    print(f"\nFake dataset created successfully!")

if __name__ == '__main__':
    create_fake_dataset()