# Training and running LV segmentation

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

import rechonet

@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None) # directory of dataset
@click.option("--output", type=click.Path(file_okay=False), default=None) # directory of outputs
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
           default="deeplabv3_resnet50") # name of segmentation model in the torchvision collection 
@click.option("--pretrained/--random", default=False) # to use pretrained or random
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None) 
@click.option("--run_test/--skip_test", default=False) # whether to run on test set
@click.option("--save_video/--skip_video", default=False) # save video 
@click.option("--num_epochs", type=int, default=50) # for training
@click.option("--lr", type=float, default=1e-5) # for SDG
@click.option("--weight_decay", type=float, default=0) # weight decay
@click.option("--lr_step_period", type=int, default=None) # period of decay
@click.option("--num_train_patients", type=int, default=None) # for ablations
@click.option("--num_workers", type=int, default=4) # subprocesses for data loading
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(data_dir=None, output=None,
        model_name="deeplabv3_resnet50", pretrained=False, weights=None,
        run_test=False, save_video=False, num_epochs=50, lr=1e-5, weight_decay=1e-5,
        lr_step_period=None, num_train_patients=None, num_workers=4,
        batch_size=20, device=None, seed=0):
    "Trains/tests segmentation model"

    # set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True) # safety

    # set device
    if device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained, aux_loss=False) # aux_loss usually disabled for inference
    
    # need to make the classifier outputs dim 1
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernal_size=model.classifier[-1].kernal_size) 
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # optim
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_period)

    # normalization
    mean, std = rechonet.utils.get_mean_and_std(rechonet.datasets.Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"] # targets for segementation
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std}
    
    # Dataset and dataloaders
    dataset = {}
    dataset["train"] = rechonet.datasets.Echo(root=data_dir, split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = rechonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Training and testing loops 
    # Do this within a log for safety
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf") # this is a good initalization
        
        # Flexibility to start from scratch or from a checkpoint 
        try:
            # Attempt to load a checkpoint (which would be saved in output)
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint['epoch'] + 1
            bestLoss = checkpoint['best_loss']
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True) # flush the stream
            for phase in ['train', 'val']: 
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase] # extract the right one
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type=="cuda"), drop_last=(phase=="train")
                )
                
                # Now we can call one epoch of segmentation
                loss, large_inter, large_union, small_inter, small_union = rechonet.utils.segmentation.run_epoch(model, dataloader, phase == "train", optimizer, device)

                # Dice calculations... these are lists. Sum reduces all dimensions
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    large_dice,
                                                                    small_dice,
                                                                    time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'best_loss': bestLoss,
                'loss': loss,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(os.path.join(output, "best.pt"))
                bestLoss = loss 

        # AFTER LOOP
        # Load best weights at the end of all epochs for validation or testing 
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            for split in ["val", "test"]:
                dataset = rechonet.datasets.Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.Dataloader(
                    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type=="cuda"))
                loss, large_inter, large_union, small_inter, small_union = rechonet.utils.segmentation.run_epoch(model, dataloader, False, None, device) # no optim for eval
                
                # Keep in mind this dice will be a list
                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                # To get a confidence interval, resample the list many times with replacement
                # Doing this at the overall, large only, and small only levels
                # Although the lists are fixed, resampling with replacement yields slightly different dices
                # The star is iterable unpacking the tuple
                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *rechonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), rechonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *rechonet.utils.bootstrap(large_inter, large_union, rechonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *rechonet.utils.bootstrap(small_inter, small_union, rechonet.utils.dice_similarity_coefficient)))
                f.flush()

    # Saving videos with segementations on top
    # Produce side by side video with original on the left, segmented on the right, and plot underneath 
    # tracking segmented region size over time, with marking systole diastole
    dataset = rechonet.datasets.Echo(root=data_dir, split="test",
                                     target_type=["Filename", "LargeIndex", "SmallIndex"],
                                     mean=mean, std=std,
                                     length=None, max_length=None, period=1)  # Take all frames 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn = _video_collate_fn)

    # IMPORTANT: video_collate pulls frames to the front

    # If every fname file does not exist ("only run if missing videos")
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        rechonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                # New output format is due to video_collate_fn
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader): 
                    
                    # seems to be the assumption that (frame, 3, w, h) 
                    y = np.concatenate([model(x[i:(i+batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        video = x[start:(start + offset), ...] # ... all other dimensions
                        logit = y[start:(start + offset), 0, :, :]    

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  
                        assert c == 3

                        # Two copies of the video side by side
                        video = np.concatenate((video, video), 3) # along w axis

                        # If pixel in segementation, saturate blue channel on the right side
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2) # along h axis

                        # Compute size of segmentation over each frame in pixel
                        size = (logit > 0).sum((1, 2)) 

                        # Identify systole frames with peak detection (negative in this case)
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0]) # need to index 0 because (peaks, properties)
                        # Distance so you don't double count, prominence defines size

                        # Write sizes and frames to file with indicators
                        # 3 flags to distinguish ground truth vs prediction
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video to draw the size over time chart with labels
                        for (f, s) in enumerate(size):
                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.
                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1)) 

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions back to "normal" and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        rechonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device): # train is a boolean, 1 is segmented
    """Modularizing one epoch of training/evaluation for segmentation."""
    total = 0.
    n = 0
    pos, neg = 0, 0
    pos_pix, neg_pix = 0, 0

    model.train(train) # optional mode=True/False -> will be eval on false

    # Label these
    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train): # dynamic 
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            # load segmentation task targets
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader: 
                # Count number of pixels in and outside human segmentations 
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # Count number of pixels in and outside computer segmentations  
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy() # axis 0?
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames and compute loss
                # Binary cross entropy loss by pixel 
                # Loss nudges the model towards right segmentation, inter/unions are pure bookkeeping for Dice 
                # Scalar vs list -> one total for whole dataset vs per sample 
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)["out"] # forward pass, output raw logits, threshold at 0 (cheaper than sigmoid)
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:,0,:,:], large_trace, reduction="sum")
                # Compute pixel intersection and union between human/computer
                large_inter += np.logical_and(y_large[:,0,:,:].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:,0,:,:].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2))) # along both axis 1 and 2?
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)["out"] # only one frame
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:,0,:,:], small_trace, reduction="sum")
                # Same logic
                small_inter += np.logical_and(y_small[:,0,:,:].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:,0,:,:].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2))) # along both axis 1 and 2?
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2)))

                # Take step with gradient if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                # probabilites?
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on pbar -> I can't even unpack this to be so honest
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    # keep in mind, these are per-video counts.
    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )

# This collate function is ideal because you don't need to pad or truncate 
def _video_collate_fn(x):
    """ Collate function for Pytorch dataloader to merge multiple videos

    Used for a dataset with video as a first element, along with some tuple of targets. Input x is a list of tuples
    where x[i][0] is the i-th video in the batch and x[i][1] are the targets for i-th video

    This returns a 3-tuple: Videos concatenated along frames dimension, targets no modification, length of videos in frames
    """

    video, target = zip(*x) # Unzipping tuples
    # video is tuple of len batch_size where each elm is (channels=3, frames, height, width)
    # targets is tuple of len batch_size where each elm is tuple of targets

    # Extracts of videos in frames -> i is a list of lengths of videos
    i = list(map(lambda t: t.shape[1], video))

    # Concatenate videos along frame dimension and pull to front
    # Essentially "playing the videos one after another"
    # Shape becomes (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, axis=1), 0, 1))
    
    target = zip(*target) # regroup/transpose a list of tuples

    return video, target, i