import random
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

def set_seed(seed):
    """Function to set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(ds='cifar10', size=0, data_dir='data', batch_size=64, num_workers=2,
              train_split=0.8, val_ratio=0.1):
    print(ds,data_dir)
    """
    Load dataset with train/val/test split and augmentation.
    Supports: CIFAR-10, CIFAR-100, ImageNet (from folders)

    Parameters:
    - ds: dataset name ('cifar10', 'cifar100', 'imagenet')
    - size: limit dataset size (for quick testing)
    - data_dir: path to dataset
    - batch_size: training batch size
    - num_workers: DataLoader workers
    - train_split: proportion of full dataset to use for training+val
    - val_ratio: fraction of train_split to allocate for validation

    Returns:
    - train_loader, val_loader, test_loader
    """

    if ds == 'cifar10':
        print("Loading CIFAR-10 dataset...")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        # Split full_trainset into train and val
        train_size = int(train_split * len(full_trainset))
        val_size = len(full_trainset) - train_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])

    elif ds == 'cifar100':
        print("Loading CIFAR-100 dataset...")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        full_trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

        train_size = int(train_split * len(full_trainset))
        val_size = len(full_trainset) - train_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])

    elif ds == 'imagenet':
        data_dir+="/imagenet-256/versions/1"
        print("Loading ImageNet dataset from folder...")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # Calculate split sizes
        total_size = len(dataset)
        trainval_size = int(train_split * total_size)
        test_size = total_size - trainval_size

        # First: split off test set
        trainval_set, testset = random_split(dataset, [trainval_size, test_size])

        # Now: split trainval into train and val
        val_size = int(val_ratio * trainval_size)
        train_size = trainval_size - val_size
        trainset, valset = random_split(trainval_set, [train_size, val_size])

        try:
            import labels
            labels.initialize_imagenet_classes(data_dir)
        except ImportError:
            print("ImageNet label initialization skipped (optional).")

    else:
        raise ValueError("Dataset not supported. Choose from 'cifar10', 'cifar100', or 'imagenet'.")

    # Optional: subsample for quick debugging
    if size > 0:
        trainset = torch.utils.data.Subset(trainset, range(min(size, len(trainset))))
        valset = torch.utils.data.Subset(valset, range(min(size, len(valset))))
        testset = torch.utils.data.Subset(testset, range(min(size, len(testset))))

    # Dataloaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

    print(f"Dataset '{ds}' loaded:")
    print(f" - Train size: {len(trainset)}")
    print(f" - Val size:   {len(valset)}")
    print(f" - Test size:  {len(testset)}")

    return train_loader, val_loader, test_loader


def visualize_image(trainloader):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    imshow(torchvision.utils.make_grid(images[:4]))

class CIFAR10C(Dataset):
    def __init__(self, corruption, severity=5, transform=None, root='../data/CIFAR-10-C'):
        self.data = np.load(f"{root}/{corruption}.npy")
        self.targets = np.load(f"{root}/labels.npy")
        self.transform = transform
        severity=severity + 1 if corruption == 'gaussian_noise' else severity
        self.data = self.data[(severity - 1) * 10000 : severity * 10000]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
def load_corrupted_cifar(corruption="gaussian_noise",severity=3):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    corrupted_dataset = CIFAR10C(corruption=corruption, severity=severity, transform=transform)
    corrupted_dataset = torch.utils.data.Subset(corrupted_dataset, range(100))
    loader = DataLoader(corrupted_dataset, batch_size=64, shuffle=False)
    return loader
