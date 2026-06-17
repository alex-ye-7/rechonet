# The EchoNet Dataset

import os
import collections
import pandas as pd

import numpy as np
import skimage.draw
import torchvision
import rechonet

class Echo(torchvision.datasets.VisionDataset): 
    def __init__(self, root=None,
                split="train", target_type="EF",
                mean=0., std=1., 
                length=16, period=2,
                max_length=250,
                clips=1,
                pad=None,
                noise=None,
                target_transform=None, # callable transform
                external_test_location=None,
                clip_contain_keyframes=False): # for frame task: window must span ED and ES
        if root is None:
            root = rechonet.config.DATA_DIR
        
        super().__init__(root, target_transform=target_transform)

        self.split = split.upper() 
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std 
        self.length = length # frames to clip
        self.max_length = max_length
        self.period = period # every period-ith frame is taken
        self.clips = clips # KEY here for test-time augmentation (inference technique that averages model on mulitple augmentations)
        self.pad = pad # for augmentation as well
        self.noise = noise # fraction of pixels to blacken
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.clip_contain_keyframes = clip_contain_keyframes
        
        self.fnames, self.outcome = [], [] # names might just ints

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pd.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            # all going to list
            if self.split != "ALL":
                data = data[data["Split"] == self.split]
            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist() 
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""] # if no extention assume avi!
            self.outcome = data.values.tolist()

            # checking for missing files
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces -> a trace is a dictionary (file) of dictionary (frame) of lists
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)
            
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")      # header
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"] # make sure everything is there

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1,y1,x2,y2))
            
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame]) # convert all to np
            
            # Videos without traces removed from dataset
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

            # Keyframe task: drop videos whose ED-ES temporal gap can't fit in a single window.
            # Window spans (length-1)*period native frames; if |ED-ES| exceeds that, no window
            # can contain both labeled keyframes, so the must-contain-both constraint is infeasible.
            if self.clip_contain_keyframes and self.length is not None:
                window_span = (self.length - 1) * self.period
                keep = [abs(self.frames[f][-1] - self.frames[f][0]) <= window_span for f in self.fnames]
                dropped = len(self.fnames) - sum(keep)
                if dropped > 0:
                    print("Clips must contain key frames: dropping {}/{} videos with |ED-ES| > {} native frames on {}".format(
                        dropped, len(self.fnames), window_span, self.split))
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        
    # to use list indexing on EchoDataset
    def __getitem__(self, index):
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        video = rechonet.utils.loadvideo(video).astype(np.float32) # (3, frames, height, width)
        
        # Add simulated noise (set some to black 0) -> why?
        # Has not been normalized yet
        # Flatten -> sample -> unflatten using division
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise*n), replace=False) # subset of positions
            f = ind % video.shape[1] # frame indicies
            ind //= video.shape[1]
            i = ind % video.shape[2] # height indicies
            ind //= video.shape[2]
            j = ind # width indicies
            video[:, f, i, j] = 0 # select their intersection

        # Normalization (pixel-wise!)
        if isinstance(self.mean, (float, int)):
            video -= self.mean 
        else: # i think this is just a check in case it's a torch/array?
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3,1,1,1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            length = f // self.period # every period-ith frame
        else:
            length = self.length # take specified # of frames
        
        if self.max_length is not None:
            length = min(length, self.max_length) # shorten to max length
    
        # Pad video with frames filled with zeros if too short
        if f < length * self.period:
            # 0 means grey now after norm
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1) # frames is pos 1
            c, f, h, w = video.shape

        # Logic for key frame task
        if self.clip_contain_keyframes:
            # Window must span both labeled keyframes. Sample start uniformly from valid range
            key = self.fnames[index]
            ed = self.frames[key][-1]
            es = self.frames[key][0]
            lo = max(0, ed - (length - 1) * self.period) # low enough to still reach ed 
            hi = min(es, f - (length - 1) * self.period - 1) # highest end would start at es
            if hi < lo: # shouldn't happen but fall back
                hi = lo
            start = np.random.randint(lo, hi + 1, size=self.clips)
        elif self.clips == "all": # Take all possible clips of desired length
            # maximum valid starting index is (f - (length - 1) * self.period) to stay within the video
            start = np.arange(f - (length-1) * self.period)
        else:
            # or pick clips from within valid starting index bounds
            start = np.random.choice(f - (length-1) * self.period, self.clips)
        
        # Gather targets, remember default is EF
        target = []
        for t in self.target_type:
            key = self.fnames[index] # remember we are using index to access
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                if self.clip_contain_keyframes:
                    # Window-relative sampled-frame index (float, in [0, length-1]).
                    target.append(np.float32((self.frames[key][-1] - start[0]) / self.period))
                else:
                    target.append(np.int32(self.frames[key][-1]))
            elif t == "SmallIndex":
                if self.clip_contain_keyframes:
                    target.append(np.float32((self.frames[key][0] - start[0]) / self.period))
                else:
                    target.append(np.int32(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1],:,:])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0],:,:])
            elif t in ["LargeTrace", "SmallTrace"]: # SEGMENTATION MASK
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                
                # Keep in mind pairwise coordinates are actually lists of coords
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

                # Build polygon boundary
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                
                # Fill in polygon
                r, c = skimage.draw.polygon(np.rint(y).astype(np.int32), np.rint(x).astype(np.int32), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32) # create blank (H,W)
                mask[r, c] = 1 # and fill the segemented LV with 1s
                target.append(mask) # that's your mask!
                
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0)) # you don't know target
                else: # otherwise 
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))
        
        # Turn target to tuple and apply transform if necessary
        if target != []: # if some target was found
            target = tuple(target) if len(target) > 1 else target[0] # depends how many
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Build clips from video based on start variable
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video) # stack them if multiple clips requested

        # Data augmentation trick to randomly shift videos -> create some noise
        if self.pad is not None:
            c, l, h, w = video.shape
            # Make a bigger video with some padding on all 4 sides
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            # Place the original video inside the padding
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            # Random crop offset! From the top left corner
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)] # now crop back

        # Now return the (3,len,h,w) with its associated tuple of targets
        return video, target 
        
    def __len__(self):
        return len(self.fnames) # how many data samples -> should be like 10k in original 
    
    def extra_repr(self) -> str: # extra
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

# To avoid issues with Windows compatability -> I wonder how they found this bug
# Needs to not be anonymous so Echo dataset can be wrapped by Dataloader
def _defaultdict_of_lists():
    return collections.defaultdict(list)