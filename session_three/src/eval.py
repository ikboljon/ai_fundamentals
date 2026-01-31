# eval.py
# Load checkpoint and run evaluation on test set. Shows MAE/RMSE and a small scatter plot.
import argparse
import torch
from model import SmallCNNRegressor
from dataset import get_loaders
from utils import get_device
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    model = SmallCNNRegressor(in_channels=3).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model

def run_eval(checkpoint, batch_size=64, img_size=64):
    device = get_device()
    _, _, test_loader = get_loaders(batch_size=batch_size, img_size=img_size)
    model = load_checkpoint(checkpoint, device)
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().ravel()
            ys.append(yb.numpy().ravel())
            ps.append(out)
    y = np.concatenate(ys); p = np.concatenate(ps)
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    print(f"Test MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    # scatter
    plt.figure(figsize=(5,4))
    plt.scatter(y, p, alpha=0.4, s=8)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    plt.plot(lims, lims, 'k--')
    plt.xlabel("True (0..4)")
    plt.ylabel("Pred")
    plt.title("True vs Pred (test)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./runs/retina/best.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()
    run_eval(args.checkpoint, batch_size=args.batch_size, img_size=args.img_size)
