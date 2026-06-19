# Training and running a model that picks end-diastolic (ED) and end-systolic (ES) frames from clip
# Regression implementation, head predicts two scalars 

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

@click.command("keyframe")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
           default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32) # just going to assume the same rate as video
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=2)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
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
        output = os.path.join("output", "keyframes", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrained flag is deprecated now
    # model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    default_w = "DEFAULT" if pretrained else None
    model = torchvision.models.video.__dict__[model_name](weights=default_w)

    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    # Bias init in middle of window for both outputs (small head-start over random init)
    model.fc.bias.data[:] = 0.5
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint["state_dict"])

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Same mean and std, and then 
    mean, std = rechonet.utils.get_mean_and_std(rechonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {
        "target_type": ["SmallIndex", "LargeIndex"], # ES first, ED second
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
        "clip_contain_keyframes": True,
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

                loss, yhat, y = run_epoch(model, dataloader, phase == "train", optim=optim, 
                                          device=device, length=frames)
                es_mae, ed_mae = mae_native(yhat, y, frames, period)

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
                loss, yhat, y = rechonet.utils.frames.run_epoch(model, dataloader, False, None, device=device, length=frames)
                es_mae, ed_mae = mae_native(yhat, y, frames, period)
                f.write("{} ES MAE {:.2f} | ED MAE {:.2f}\n".format(split, es_mae, ed_mae))

                yhat_f = yhat * (frames - 1) * period   # native-frame positions
                y_f  = y  * (frames - 1) * period
                for k, name in [(0, "ES"), (1, "ED")]:
                    plt.scatter(y_f[:, k], yhat_f[:, k], s=2)   # true vs predicted
                    plt.plot([y_f[:,k].min(), y_f[:,k].max()], [y_f[:,k].min(), y_f[:,k].max()])  # y=x
                    plt.savefig(os.path.join(output, "{}_{}.pdf".format(split, name)))
                    plt.clf()


def run_epoch(model, dataloader, train, optim, device, length=32):
    """One epoch of train/eval for the keyframe regressor."""
    model.train(train)

    total_loss = 0.
    n = 0
    yhat, y = [], []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, targets) in dataloader:
                # targets is a tuple (es_idx, ed_idx), each tensor of shape (B,), in sampled-frame units
                es_idx, ed_idx = targets
                # Normalize to [0, 1] across the sampled window, as length and period could change
                yb = torch.stack([es_idx.float() / (length-1), 
                                 ed_idx.float() / (length-1)], dim=1).to(device) # (B, 2)
                X = X.to(device)

                yhat_b = model(X) # (B, 2)
                loss = torch.nn.functional.smooth_l1_loss(yhat_b, yb)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                yhat.append(yhat_b.to("cpu").detach().numpy())
                y.append(yb.to("cpu").detach().numpy())

                total_loss += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("loss {:.4f}".format(total_loss / n))
                pbar.update()

    return total_loss / n, np.concatenate(yhat), np.concatenate(y)

def mae_native(yhat, y, length, period):
    err = np.abs(yhat - y) * (length - 1) * period   # back to native frames
    return err[:, 0].mean(), err[:, 1].mean()         # (ES_MAE, ED_MAE)

if __name__ == "__main__":
    run()
