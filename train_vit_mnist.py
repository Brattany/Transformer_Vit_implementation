import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.vit import VisionTransformer

# 加载 MNIST 数据集，并划分为 train / val / test
def get_data_loaders(data_dir, batch_size, num_workers, val_size=10000, seed=42):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),  # 随机裁剪并填充
        transforms.RandomRotation(10),  # 随机旋转 ±10 度
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform
    )

    # MNIST 官方测试集：10000 张
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )

    # 从官方训练集中划分出验证集
    assert 0 < val_size < len(train_dataset), f"val_size must be between 1 and {len(train_dataset) - 1}, but got {val_size}"
    train_size = len(train_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, val_loader, test_loader

#训练一个epoch
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

#创建vit模型
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


#读取命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on MNIST.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser.parse_args()

def set_seed(seed):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    assert 28 % args.patch_size == 0, "patch_size must divide img_size=28"
    assert args.embed_dim % args.num_heads == 0, "embed_dim must be divisible by num_heads"
    
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=args.seed
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    model = build_model(args).to(device)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    #学习率衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    epoch_times = []
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=eval_criterion,
            device=device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        print(
            f"Epoch {epoch} summary: "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% "
            f"epoch_time={epoch_time:.2f}s avg_epoch_time={avg_epoch_time:.2f}s"
        )

        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "args": vars(args)
                },
                save_path
            )
            print(f"Saved best checkpoint to {save_path} with acc={best_val_acc * 100:.2f}%")
        else:
            epochs_no_improve += 1
            print(
                f"No improvement for {epochs_no_improve}/{args.patience} epochs. "
                f"Best val_acc={best_val_acc * 100:.2f}%"
            )

            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        
        scheduler.step()#更新学习率
    
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=eval_criterion,
            device=device
        )

        print(
            f"Final test result: "
            f"test_loss={test_loss:.4f} test_acc={test_acc * 100:.2f}%"
        )

    if epoch_times:
        print(f"Average epoch time over {len(epoch_times)} epochs: {sum(epoch_times) / len(epoch_times):.2f}s")


if __name__ == "__main__":
    main()