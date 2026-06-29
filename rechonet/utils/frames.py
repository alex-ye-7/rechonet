# Model that picks end-diastolic (ED) and end-systolic (ES) frames direct from video

import math
import os
import time

import click
import numpy as np
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt

import rechonet

# Custom ResNet + LSTM 
class FrameNet(torch.nn.Module):
    def __init__(self, backbone="resnet18", temporal="lstm", pretrained=True, hidden=128):
        super().__init__()
        enc = torchvision.models.__dict__[backbone](weights="DEFAULT" if pretrained else None)
        self.feat_dim = enc.fc.in_features
        enc.fc = torch.nn.Identity() # remove classification head -> turn temporal
        self.enc = enc # Store the backbone

        if temporal == "lstm":
            self.temporal = torch.nn.LSTM(self.feat_dim, hidden, batch_first=True, bidirectional=True)
            head_in = 2 * hidden # bidirectional doubles output dimension (concatenates forward/backward)
        elif temporal == 'tcn':
            self.temporal = torch.nn.Sequential(
                torch.nn.Conv1d(self.feat_dim, hidden, kernel_size=3, padding=1, dilation=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
                torch.nn.ReLU()
            )
            head_in = hidden
        else:
            raise ValueError("temporal arch must be lstm or tcn, got {}".format(temporal))

        self.head = torch.nn.Conv1d(head_in, 2, kernel_size=1) # fully connected layer, two values

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W) # each frame is a seperate image, combine batch and time
        feats = self.enc(x).view(B, T, self.feat_dim) # (B, T, F) 

        if self.temporal == "lstm": # LSTM expects (B, T, feat_dim)
            seq, _ = self.temporal(feats) # (B, T, 2H)
            seq = seq.transpose(1,2) # (B, 2H, T)
        else: # Conv1d expects (B, C/feat_dim, T/length)
            seq = self.temporal(feats.transpose(1,2)) # (B,H,T)

        logits = self.head(seq) # (B, 2, T)
        return logits.transpose(1, 2) # (B, T, 2) - two values per frame

# Helper functions

# yhat and y are sampled-frame indicies, need to multiply by period
def mae_native(yhat, y, period):
    err = np.abs(yhat - y) * period   # back to native frames
    return err[:, 0].mean(), err[:, 1].mean()        

# both (B,T)
# Target is a Gaussian shape but not distribution yet
def soft_ce_loss(logits, target): 
    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
    logp = torch.nn.functional.log_softmax(logits, dim=1) # across time
    return -(target * logp).sum(dim=1).mean()

# turn (B,T) -> (B,)
# Expected value formula: find weighted average frame over distribution
def soft_argmax(logits):
    T = logits.shape[1]
    prob = torch.softmax(logits, dim=1)
    pos = torch.arange(T, device=logits.device, dtype=prob.dtype)
    return (prob * pos).sum(dim=1)
    

# CLI
@click.command("keyframe")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--backbone", type=str, default="resnet18")
@click.option("--temporal", type=click.Choice(["lstm", "tcn"]), default="lstm")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32) 
@click.option("--period", type=int, default=2)
@click.option("--sigma", type=float, default=1.5)
@click.option("--hidden", type=int, default=128)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=2)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    backbone="resnet18",
    temporal="lstm",
    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=45,
    lr=1e-3,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    sigma=1.5,
    hidden=128,
    num_train_patients=None,
    num_workers=2,
    batch_size=20,
    device=None,
    seed=0,
):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default output dir
    if output is None:
        output = os.path.join("output", "keyframes", "{}_{}_{}_{}".format(backbone, temporal, frames, period))
    os.makedirs(output, exist_ok=True)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrameNet(backbone=backbone, temporal=temporal, pretrained=pretrained, hidden=hidden)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint["state_dict"])

    #optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    mean, std = rechonet.utils.get_mean_and_std(rechonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {
        "target_type": ["SmallHeatmap", "LargeHeatmap", "SmallIndex", "LargeIndex"], # ES first, ED second
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
        "clip_contain_keyframes": True,
        "heatmap_std": sigma,
    }

    dataset = {}
    dataset["train"] = rechonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = rechonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Training and testing loop
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["opt_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ["train", "val"]:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                    pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, yhat, y = rechonet.utils.frames.run_epoch(model, dataloader, phase == "train", optim=optim, device=device)
                es_mae, ed_mae = mae_native(yhat, y, period)

                f.write("{},{},{},{},{},{}\n".format(
                    epoch, phase, loss, es_mae, ed_mae, time.time() - start_time))
                f.flush()

            scheduler.step()

            # Save best by val loss
            save = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                "opt_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                bestLoss = loss
                torch.save(save, os.path.join(output, "best.pt"))

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            for split in ["val", "test"]:
                # Without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    rechonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = rechonet.utils.frames.run_epoch(model, dataloader, False, None, device=device)
                es_mae, ed_mae = mae_native(yhat, y, period)
                f.write("{} ES MAE {:.2f} | ED MAE {:.2f}\n".format(split, es_mae, ed_mae))
                f.flush()
                print("{} ES MAE {:.2f} | ED MAE {:.2f}\n".format(split, es_mae, ed_mae))
                yhat_f = yhat * period   # native-frame positions
                y_f  = y * period
                for k, name in [(0, "ES"), (1, "ED")]:
                    plt.scatter(y_f[:, k], yhat_f[:, k], s=2)   # true vs predicted
                    plt.plot([y_f[:,k].min(), y_f[:,k].max()], [y_f[:,k].min(), y_f[:,k].max()])  # y=x
                    plt.xlabel("true frame")
                    plt.ylabel("predicted frame")
                    plt.savefig(os.path.join(output, "{}_{}.pdf".format(split, name)))
                    plt.clf()


def run_epoch(model, dataloader, train, optim, device):
    """
    One epoch of train/eval for the keyframe regressor
    Returns (avg_loss, yhat, y) where (N,2) arrays in sampled frames 
    """
    model.train(train)

    total_loss = 0.
    n = 0
    yhat, y = [], []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, targets) in dataloader:
                es_hm, ed_hm, es_idx, ed_idx = targets # (B,T), (B,T), (B,), (B,)
                
                X = X.to(device)
                logits = model(X) # (B,T,2)
                loss = soft_ce_loss(logits[...,0], es_hm.to(device)) + soft_ce_loss(logits[...,1], ed_hm.to(device))

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Extract predicted frame from distribution
                pred_es = soft_argmax(logits[...,0]).to("cpu").detach().numpy()
                pred_ed = soft_argmax(logits[...,1]).to("cpu").detach().numpy()

                # Build up the eventual concat, ensure es/ed are saved together for one sample
                yhat.append(np.stack([pred_es, pred_ed], axis=1))
                y.append(np.stack([es_idx.numpy(), ed_idx.numpy()], axis=1))

                total_loss += loss.item() * X.size(0)
                n += X.size(0)
                pbar.set_postfix_str("loss {:.4f}".format(total_loss / n))
                pbar.update()

    return total_loss / n, np.concatenate(yhat), np.concatenate(y)


if __name__ == "__main__":
    run()
