import os
import sys
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from utils import CombinedLoss, compute_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        metrics = compute_metrics(outputs, masks)
        running_dice += metrics["dice"] * images.size(0)
        running_iou += metrics["iou"] * images.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{metrics['dice']:.4f}"})

    total_samples = len(dataloader.dataset)
    return {
        "loss": running_loss / total_samples,
        "dice": running_dice / total_samples,
        "iou": running_iou / total_samples
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            metrics = compute_metrics(outputs, masks)
            running_dice += metrics["dice"] * images.size(0)
            running_iou += metrics["iou"] * images.size(0)

    total_samples = len(dataloader.dataset)
    return {
        "loss": running_loss / total_samples,
        "dice": running_dice / total_samples,
        "iou": running_iou / total_samples
    }


def main(config_path="configs/config.yaml"):
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set device
    device_name = cfg["project"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Build Dataloaders
    print("Preparing Datasets...")
    try:
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure images and masks exist in the directories configured in config.yaml.")
        return

    # Build Model
    model = build_model(cfg).to(device)

    # Optimizer & Loss
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 1e-5))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = int(cfg["training"]["epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg["training"].get("min_lr", 1e-6)))
    criterion = CombinedLoss()

    # Save directory
    save_dir = cfg["training"].get("save_dir", "./weights")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_val_dice = 0.0
    patience = int(cfg["training"].get("early_stopping_patience", 10))
    patience_counter = 0

    print(f"\n--- Starting Training for {epochs} Epochs ---")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        # Checkpoint saving
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "config": cfg
            }, best_model_path)
            print(f"  --> Saved new best model checkpoint (Val Dice: {best_val_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nTraining completed! Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Model saved at: {best_model_path}")


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(cfg_file)
