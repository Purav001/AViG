# Path: prepare_cifar10.py
import os
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image

def save_cifar10_as_folders(root='./cifar10_dataset', train=True):
    """Downloads CIFAR10 and saves it in the ImageFolder structure."""
    
    split = 'train' if train else 'val'
    dataset_path = os.path.join(root, split)

    # Download the dataset
    cifar_dataset = CIFAR10(root=root, train=train, download=True)
    
    print(f"Preparing '{split}' split...")

    # Create the class folders
    class_names = cifar_dataset.classes
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        os.makedirs(class_path, exist_ok=True)

    # Save each image to its corresponding class folder
    for i, (image_tensor, label_index) in enumerate(cifar_dataset):
        class_name = class_names[label_index]
        image_path = os.path.join(dataset_path, class_name, f"{i}.png")
        
        # The dataset provides a PIL image directly, so we can just save it
        image_tensor.save(image_path)
        
        if (i + 1) % 5000 == 0:
            print(f"  ...saved {i+1} images")
    
    print(f"Finished preparing the {split} split in: {dataset_path}")

if __name__ == '__main__':
    # Create the main dataset directory
    if not os.path.exists('./cifar10_dataset'):
        os.makedirs('./cifar10_dataset')

    # Prepare the training set
    save_cifar10_as_folders(train=True)
    
    # Prepare the validation/test set
    save_cifar10_as_folders(train=False)

    print("\nCIFAR-10 dataset is ready in the 'cifar10_dataset' folder!")