import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vit import VisionTransformer


def get_data_loaders(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               #清空梯度
        logits = model(images)              #前向传播
        loss = criterion(logits, labels)    #计算损失
        loss.backward()                     #反向传播   
        optimizer.step()                    #更新参数

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if step % 100 == 0 or step == len(loader):
            avg_loss = total_loss / total
            acc = correct / total * 100
            print(f"Epoch {epoch} [{step:04d}/{len(loader)}] loss={avg_loss:.4f} acc={acc:.2f}%")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def build_model(args):
    return VisionTransformer(
        img_size=28,
        patch_size=args.patch_size,
        in_chans=1,
        embed_dim=args.embed_dim,
        num_classes=10,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on MNIST.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_acc = 0.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch {epoch} summary: "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc * 100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "args": vars(args)
                },
                save_path
            )
            print(f"Saved best checkpoint to {save_path} with acc={best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
