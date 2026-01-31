# train.py
# Training script with argparse, checkpointing and simple logging.
import argparse
from dataset import get_loaders
from model import SmallCNNRegressor
from utils import set_seed, get_device, save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().ravel()
            ys.append(yb.numpy().ravel())
            preds.append(out)
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    return {'mae': mae, 'rmse': rmse, 'y': y, 'p': p}

def train(args):
    set_seed(args.seed)
    device = get_device()
    writer = SummaryWriter(log_dir=args.model_dir) if args.use_tensorboard else None

    train_loader, val_loader, test_loader = get_loaders(batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    model = SmallCNNRegressor(in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_mae = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses)})
        # validation
        val_metrics = evaluate(model, val_loader, device)
        val_mae = val_metrics['mae']; val_rmse = val_metrics['rmse']
        print(f"Epoch {epoch}  val_mae={val_mae:.4f}  val_rmse={val_rmse:.4f}")
        if writer:
            writer.add_scalar('val/mae', val_mae, epoch)
            writer.add_scalar('val/rmse', val_rmse, epoch)
            writer.add_scalar('train/loss', np.mean(losses), epoch)
        # checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_mae': val_mae
            }, f"{args.model_dir}/best.pth")
    if writer: writer.close()
    # final test evaluation using best model
    print("Training finished. Best val MAE:", best_val_mae)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_dir", type=str, default="./runs/retina")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action='store_true', default=False)
    args = parser.parse_args()
    train(args)
