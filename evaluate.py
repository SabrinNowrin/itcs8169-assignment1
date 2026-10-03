"""Evaluate a saved checkpoint on the test set.

Usage:
    python evaluate.py --checkpoint checkpoints/final_resnet18.pt --data_root data
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from train import get_eval_transform, make_resnet18


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', default='data', help='folder containing test/')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = datasets.ImageFolder(Path(args.data_root) / 'test', transform=get_eval_transform(args.img_size))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    classes = test_dataset.classes

    model = make_resnet18(len(classes), pretrained=False)  # weights come from the checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    correct = [0] * len(classes)
    total = [0] * len(classes)
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        for p, y in zip(preds, labels):
            total[y] += 1
            correct[y] += int(p == y)

    print(f'Test images: {sum(total)}')
    print(f'TEST ACCURACY: {sum(correct) / sum(total):.4f}')
    print('Per-class accuracy (hardest first):')
    for name, c, t in sorted(zip(classes, correct, total), key=lambda x: x[1] / x[2]):
        print(f'  {name:<15s} {c / t:.3f}')


if __name__ == '__main__':
    main()
