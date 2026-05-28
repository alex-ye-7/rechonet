# Training and running EF prediction

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import rechonet
import rechonet.datasets
import rechonet.utils
import rechonet.utils.video


@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None) # directory of dataset
@click.option("--output", type=click.Path(file_okay=False), default=None) # output/video/<model_name>_<pretrained/random>/.
@click.option("--task", type=str, default="EF") # Options are the headers of FileList.csv
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
           default="r2plus1d_18") # video instead of segmentation model
@click.option("--pretrained/--random", default=False) # to use pretrained or random
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None) 
@click.option("--run_test/--skip_test", default=False) # whether to run on test set
@click.option("--num_epochs", type=int, default=50) # for training
@click.option("--lr", type=float, default=1e-5) # for SDG
@click.option("--weight_decay", type=float, default=0) # weight decay
@click.option("--lr_step_period", type=int, default=None) # period of decay
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None) # for ablations
@click.option("--num_workers", type=int, default=4) # subprocesses for data loading
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    task="EF",

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
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default output dict
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model

    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    # Mess with the fully conncted layer
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6 # well this seems random
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period == None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Mean and std still
    mean, std = rechonet.utils.get_mean_and_std(rechonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
            }

    # Dataset and dataloaders 
    # Need a train, val, and test dataset -> seems the splits are already done
    dataset = {}
    dataset["train"] = rechonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12) # pad for robustness
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Need to subsample
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = rechonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Training and testing
    # Open with a appends and preserves existing
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try: 
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch{}\n".format(epoch_resume))
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
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                
                # Now do the loop
                loss, yhat, y = rechonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat), # true, pred
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            
            # Ultimately we want to test on the best loss model
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()
        
        if run_test:
            for split in ["val", "test"]:
                # Without test-time augmentation
                dataset = rechonet.datasets.Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                
                loss, yhat, y = rechonet.utils.video.run_epoch(model, dataloader, False, optim, device)
                
                # Bootstrap every metric for confidence intereval
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *rechonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *rechonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, rechonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # With test-time augmentation
                ds = rechonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                # ahh test time is when the save and batch_size args are changed
                loss, yhat, y = rechonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)                
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *rechonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *rechonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, rechonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()       

                # Log
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                rechonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)



def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """One epoch of train/eval 

    train (bool)
    save_all (bool) True returns augmentation predictions seperately, False returns mean prediction 
    block_size (int or None) maximum number of augmentations?
    """

    model.train(train)

    total = 0 # total training loss
    n = 0 # n videos
    s1 = 0 # sum of ground truth EF
    s2 = 0 # sum of ground truth EF squared

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader: # outcome is a list of scalars
                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6) # check if there is a clips dimension for EF averaging 
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w) # condense batch and clips?
                
                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                # forward prop
                if block_size is None: # how many frames to index
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j+block_size), ...]) for j in range(0, X.shape[0], block_size)])

                # outputs.view(-1) flatten a tensor to 1D
                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)
                    
                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())
                
                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome) 

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                
                # either way X.size(0) is just a batch size
                total += loss.item() * X.size(0)
                n += X.size(0)

                # Running average loss for this epoch, loss for current batch, variance of outcomes
                # Printing the variance is baseline (MSE of mean-predicting model is variance)
                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    # put all y and yhat together by batch
    if not save_all:
        yhat = np.concatenate(yhat) 
    y = np.concatenate(y) # 

    return total / n, yhat, y 

if __name__ == '__main__':
    run()