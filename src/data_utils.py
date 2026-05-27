import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import random
import torch
from datasets import load_dataset
import os
import urllib.request
import zipfile
from PIL import Image


def get_cifar10_transforms():
    transform_train = transforms.Compose([
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test

def get_cifar10_dataset(transform_train, transform_test, root="./data", validation_split=0.2):
    # Create the full training dataset with training transforms
    full_train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    
    # Split the training data
    train_size = len(full_train_dataset)
    validation_size = int(train_size * validation_split)
    train_size = train_size - validation_size
    
    train_indices, validation_indices = torch.utils.data.random_split(
        list(range(len(full_train_dataset))), [train_size, validation_size]
    )
    
    # Create train dataset with training transforms
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    
    # Create validation dataset with test transforms (no augmentation)
    validation_dataset_raw = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_test)
    validation_dataset = torch.utils.data.Subset(validation_dataset_raw, validation_indices.indices)
    
    return train_dataset, validation_dataset, test_dataset

def get_cifar10_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, validation_loader, test_loader

def get_client_subsets(train_dataset, num_clients=10):
    train_size = len(train_dataset)
    indices = list(range(train_size))
    random.shuffle(indices)
    subset_size = train_size // num_clients
    client_subsets = []
    for i in range(num_clients):
        start = i * subset_size
        end = (i + 1) * subset_size if i < num_clients - 1 else train_size
        client_subsets.append(
            torch.utils.data.Subset(train_dataset, indices[start:end])
        )   
    return client_subsets

def get_client_loaders(client_datasets, batch_size=128):
    client_loaders = []
    for client_dataset in client_datasets:
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        client_loaders.append(client_loader)
    return client_loaders

def get_test_loader(test_dataset, batch_size=128):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loader

def get_wikitext2_dataloader(tokenizer, batch_size=128, seq_length=128):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=seq_length)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_dataset = tokenized_datasets["train"]
    validation_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader

def get_cifar100_transforms():
    transform_train = transforms.Compose([
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return transform_train, transform_test

def get_cifar100_dataset(transform_train, transform_test, root="./data", validation_split=0.2):
    # Create the full training dataset with training transforms
    full_train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    
    # Split the training data
    train_size = len(full_train_dataset)
    validation_size = int(train_size * validation_split)
    train_size = train_size - validation_size
    
    train_indices, validation_indices = torch.utils.data.random_split(
        list(range(len(full_train_dataset))), [train_size, validation_size]
    )
    
    # Create train dataset with training transforms
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    
    # Create validation dataset with test transforms (no augmentation)
    validation_dataset_raw = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_test)
    validation_dataset = torch.utils.data.Subset(validation_dataset_raw, validation_indices.indices)
    
    return train_dataset, validation_dataset, test_dataset

def get_cifar100_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, validation_loader, test_loader

def get_medqa_dataloader(tokenizer, batch_size=128, eval_batch_size=None, seq_length=512):
    # Use eval_batch_size for validation/test if provided, otherwise use batch_size
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Use the GBaker/MedQA-USMLE-4-options dataset which has the correct format
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    
    # We don't tokenize here - keep the raw QA format for the accuracy evaluation
    # The dataset already has separate train and test splits
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Create a validation split from train data (20% of train)
    train_size = len(train_data)
    val_size = int(train_size * 0.2)
    train_indices = list(range(train_size - val_size))
    val_indices = list(range(train_size - val_size, train_size))
    
    validation_data = train_data.select(val_indices)
    train_data = train_data.select(train_indices)
    
    # Use different batch sizes for training vs evaluation
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    validation_loader = DataLoader(validation_data, batch_size=eval_batch_size, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, collate_fn=lambda x: x)
    
    return train_loader, validation_loader, test_loader


class TinyImageNetDataset(torch.utils.data.Dataset):
    """Custom Dataset for TinyImageNet"""
    
    def __init__(self, root, split='train', transform=None, download=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, 'tiny-imagenet-200')
        
        if download:
            self._download()
        
        if not os.path.exists(self.data_dir):
            raise RuntimeError('TinyImageNet not found. Set download=True to download it.')
        
        self._load_data()
    
    def _download(self):
        """Download TinyImageNet dataset"""
        if os.path.exists(self.data_dir):
            return
        
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(self.root, 'tiny-imagenet-200.zip')
        
        print(f'Downloading TinyImageNet from {url}')
        os.makedirs(self.root, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        
        print('Extracting TinyImageNet...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        os.remove(zip_path)
        print('TinyImageNet download complete.')
    
    def _load_data(self):
        """Load data paths and labels"""
        self.images = []
        self.labels = []
        
        # Load class names
        class_names_file = os.path.join(self.data_dir, 'wnids.txt')
        with open(class_names_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        if self.split == 'train':
            train_dir = os.path.join(self.data_dir, 'train')
            for class_name in self.class_names:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            self.images.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.class_to_idx[class_name])
        
        elif self.split in ['val', 'test']:
            val_dir = os.path.join(self.data_dir, 'val')
            
            # Load validation annotations
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            val_img_to_class = {}
            
            with open(val_annotations, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    val_img_to_class[parts[0]] = parts[1]
            
            val_images_dir = os.path.join(val_dir, 'images')
            for img_name in os.listdir(val_images_dir):
                if img_name.endswith('.JPEG') and img_name in val_img_to_class:
                    self.images.append(os.path.join(val_images_dir, img_name))
                    class_name = val_img_to_class[img_name]
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_tinyimagenet_transforms():
    """Get transforms for TinyImageNet with ImageNet normalization"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform_train, transform_test


def get_tinyimagenet_dataset(transform_train, transform_test, root="./data", validation_split=0.2):
    """Get TinyImageNet datasets with train/validation/test splits"""
    
    # Load train and validation datasets
    full_train_dataset = TinyImageNetDataset(root=root, split='train', 
                                           transform=transform_train, download=True)
    val_dataset = TinyImageNetDataset(root=root, split='val', 
                                    transform=transform_test, download=False)
    
    # Split training data for training and validation
    train_size = len(full_train_dataset)
    validation_size = int(train_size * validation_split)
    train_size = train_size - validation_size
    
    train_indices, validation_indices = torch.utils.data.random_split(
        list(range(len(full_train_dataset))), [train_size, validation_size]
    )
    
    # Create train dataset with training transforms
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    
    # Create validation dataset with test transforms (no augmentation)
    validation_dataset_raw = TinyImageNetDataset(root=root, split='train', 
                                                transform=transform_test, download=False)
    validation_dataset = torch.utils.data.Subset(validation_dataset_raw, validation_indices.indices)
    
    # Use original validation set as test set
    test_dataset = val_dataset
    
    return train_dataset, validation_dataset, test_dataset


def get_tinyimagenet_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size=128):
    """Get TinyImageNet data loaders"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, validation_loader, test_loader