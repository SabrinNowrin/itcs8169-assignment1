"""Train the final model: ResNet-18 pretrained on ImageNet-1K, all layers fine-tuned.

Usage:
    python train.py --data_root data
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet normalization


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_eval_transform(img_size):
    # Validation/test photos: no random changes, so the score is fair
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def make_resnet18(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # pretrained on ImageNet-1K
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new last layer: 1000 -> 16 classes
    return model


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='folder containing train/ and test/')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='checkpoints/train_py_resnet18.pt')
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| PyTorch', torch.__version__)

    # Training photos: random changes every time they're seen
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Load the training folder twice: augmented copy for training, clean copy for validation
    train_root = Path(args.data_root) / 'train'
    train_base = datasets.ImageFolder(train_root, transform=train_transform)
    val_base = datasets.ImageFolder(train_root, transform=get_eval_transform(args.img_size))
    num_classes = len(train_base.classes)

    val_size = int(round(len(train_base) * 0.20))
    train_size = len(train_base) - val_size
    train_dataset, _ = random_split(train_base, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    _, val_dataset = random_split(val_base, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(args.seed))
    print(f'Classes: {num_classes} | Train: {train_size} | Validation: {val_size}')

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    set_random_seed(args.seed)
    model = make_resnet18(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, copy.deepcopy(model.state_dict())
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_seen += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:  # keep the epoch with the best validation accuracy
            best_val_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch:02d}/{args.epochs} | train loss {total_loss / total_seen:.4f} | '
              f'val loss {val_loss:.4f} | val acc {val_acc:.4f} | {time.time() - start:.1f}s')

    print(f'Best validation accuracy: {best_val_acc:.4f}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    print('Saved checkpoint:', args.out)


if __name__ == '__main__':
    main()
