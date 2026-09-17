import torchvision.transforms as T

def get_train_transforms(image_size):
    """
    Medical image-specific augmentations for training.
    """
    return T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # Normalize to [-1, 1] for VQ-VAE
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_inference_transforms():
    """
    Transforms for inference/triage without augmentations.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
