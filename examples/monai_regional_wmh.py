"""Using the parcellation masks downstream in MONAI (e.g. as extra channels / targets).

Given the per-region masks written by `wmhparc parcellate --split-masks`, build a MONAI
dataset that loads FLAIR + WMH + the region-labelled WMH map and computes per-region
volumes on the fly.  Requires `pip install monai`.
"""
import glob

import numpy as np
import torch
from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, MapTransform

from wmhparc.atlases import load_lut


class RegionalVolumesd(MapTransform):
    """Add a vector of regional WMH volumes (ml) from a label-coded WMH map."""

    def __init__(self, keys, lut_name="bullseye36", out_key="regional_volumes"):
        super().__init__(keys)
        self.lut = load_lut(lut_name)
        self.out_key = out_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            vox_ml = float(np.prod(img.pixdim[:3])) / 1000.0 if hasattr(img, "pixdim") else 1e-3
            arr = np.asarray(img).squeeze()
            vols = [float((arr == idx).sum()) * vox_ml for idx in self.lut["index"]]
            d[self.out_key] = torch.tensor(vols, dtype=torch.float32)
        return d


files = sorted(glob.glob("derivatives/wmhparc/sub-*/sub-*_atlas-bullseye36_space-native_desc-wmhparc_dseg.nii.gz"))
items = [{"wmhparc": f} for f in files]
xform = Compose([LoadImaged(keys=["wmhparc"]), EnsureChannelFirstd(keys=["wmhparc"]),
                 RegionalVolumesd(keys=["wmhparc"], lut_name="bullseye36")])
ds = Dataset(items, transform=xform)
if len(ds):
    print(ds[0]["regional_volumes"])
