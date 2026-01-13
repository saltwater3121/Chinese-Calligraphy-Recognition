from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def build_loaders(root: str, transform, batch_size: int, num_workers: int):
    train_ds = ImageFolder(f"{root}/train", transform=transform)
    val_ds = ImageFolder(f"{root}/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.class_to_idx
