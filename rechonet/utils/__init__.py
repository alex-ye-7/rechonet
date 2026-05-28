# Utility functions 

import os
import typing

import cv2 
import matplotlib
import numpy as np
import torch
import tqdm

from . import video
from . import segmentation
from . import frames


# Computes mean and std from samples from a Pytorch dataset (per channel)

# dataset[i][0] is expected to be the i-th video in the dataset
# should be a Tensor of (channels=3, frames, height, width)
def get_mean_and_std(dataset: torch.utils.data.Dataset, samples: int = 128, 
                     batch_size: int = 8, num_workers: int = 4):
    if samples is not None and len(dataset) > samples: # sample from the dataset
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elms
    s1 = 0.  # sum of elements along channels (ends up np.array dim (channels,) )
    s2 = 0.  # sum of squares of elements along channels (same dim as above)

    for x, *_ in tqdm.tqdm(dataloader): # x is (batch, 3, frames, h, w)
        # Flatten every pixel from every frame of every video in the batch, grouped by channel
        x = x.transpose(0,1).contiguous().view(3, -1) # bring 3 channels to front
        n += x.shape[1] # 1 position becomes batch*h*w
        s1 += torch.sum(x, dim=1).numpy() # collapse along dim 1 and numpy
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n 
    std = np.sqrt(s2 / n - mean ** 2)

    # just in case
    mean = mean.astype(np.float32) 
    std = std.astype(np.float32)
    return mean, std


# Filename of video and loads into into np array of (channels=3, frames, height, width)
# Values are ints ranging from 0 to 255
def loadvideo(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError
    capture = cv2.VideoCapture(filename)

    # are this video capture properties still in use?
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_width, frame_height, 3)) # not sure why doing this then transpose later?

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame
    v = v.transpose((3,0,1,2))
    return v

# Saves a video to a file
# Cool to see how a video is constructed from first principles
def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    
    c, _, height, width = array.shape 
    if c != 3: 
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape)))) 

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in array.transpose((1,2,3,0)): # move channels to the back?
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)


# Compites a bootstrapped confidence intervals for func(a,b)
# Returns a tuple of (func(a, b), estimated 5-th percentile, estimated 95-th percentile)
def bootstrap(a, b, func, samples=10000):
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(b))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps) # order them for 0.05 and 0.95 extraction

    return func(a,b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))

# Defines the module's public API
__all__ = ["video", "segmentation", "frames", "loadvideo", "savevideo", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]