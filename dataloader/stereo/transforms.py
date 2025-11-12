from __future__ import division
import torch
import numpy as np
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        if self.no_normalize:
            sample['left'] = torch.from_numpy(left)
        else:
            sample['left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['right'], (2, 0, 1))

        if self.no_normalize:
            sample['right'] = torch.from_numpy(right)
        else:
            sample['right'] = torch.from_numpy(right) / 255.

        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)
        if 'disp_r' in sample.keys():
            disp_r = sample['disp_r']  # [H, W]
            sample['disp_r'] = torch.from_numpy(disp_r)

        if 'valid' in sample.keys():
            valid = sample['valid']  # [H, W]
            sample['valid'] = torch.from_numpy(valid)

        return sample


class Resize(object):
    def __init__(self,
                 scale_x=1,
                 scale_y=1,
                 nearest_interp=True,  # for sparse gt
                 ):
        """
        Resize low-resolution data to high-res for mixed dataset training
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.nearest_interp = nearest_interp

    def __call__(self, sample):
        scale_x = self.scale_x
        scale_y = self.scale_y

        sample['left'] = cv2.resize(sample['left'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        sample['right'] = cv2.resize(sample['right'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if 'disp' in sample.keys():
            sample['disp'] = cv2.resize(
                sample['disp'], None, fx=scale_x, fy=scale_y,
                interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
            ) * scale_x

        if 'disp_r' in sample.keys():
            sample['disp_r'] = cv2.resize(
                sample['disp_r'], None, fx=scale_x, fy=scale_y,
                interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
            ) * scale_x

        return sample