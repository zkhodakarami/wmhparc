import numpy as np
import nibabel as nib
import pandas as pd

from wmhparc.imageio import resample_to_grid, save_labels
from wmhparc.parcellate import fill_unlabelled, parcellate_wmh, regional_volumes


def test_regional_volumes_counts():
    atlas = np.zeros((10, 10, 10), np.int16)
    atlas[:5] = 1
    atlas[5:8] = 2
    wmh = np.zeros_like(atlas, bool)
    wmh[0:2, 0, 0] = True   # 2 voxels in region 1
    wmh[6, 0, 0] = True     # 1 voxel in region 2
    wmh[9, 0, 0] = True     # 1 unlabelled
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    lut = pd.DataFrame({"index": [1, 2], "name": ["a", "b"]})
    df = regional_volumes(wmh, atlas, aff, lut)
    assert list(df["wmh_voxels"]) == [2, 1, 1]
    assert np.isclose(df.loc[0, "wmh_volume_ml"], 2 * 0.008)
    assert np.isclose(df.loc[0, "region_volume_ml"], 500 * 0.008)
    assert np.isclose(df["fraction_of_total_wmh"].sum(), 1.0)
    assert df.attrs["total_wmh_voxels"] == 4


def test_fill_unlabelled_nearest():
    atlas = np.zeros((20, 5, 5), np.int16)
    atlas[:5] = 1
    atlas[15:] = 2
    wmh = np.zeros_like(atlas, bool)
    wmh[6, 2, 2] = True    # 2 mm from region 1
    wmh[13, 2, 2] = True   # 2 mm from region 2
    wmh[10, 2, 2] = True   # 5 mm from both
    filled = fill_unlabelled(wmh, atlas, np.eye(4), max_mm=3)
    assert filled[6, 2, 2] == 1 and filled[13, 2, 2] == 2 and filled[10, 2, 2] == 0
    assert (filled != atlas).sum() == 2


def test_resample_to_grid_identity_and_downsample():
    data = np.arange(8 * 8 * 8, dtype=np.int32).reshape(8, 8, 8)
    aff = np.eye(4)
    out = resample_to_grid(data, aff, aff, data.shape, order=0)
    assert np.array_equal(out, data)
    aff2 = np.diag([2.0, 2.0, 2.0, 1.0])
    out2 = resample_to_grid(data, aff, aff2, (4, 4, 4), order=0)
    assert out2.shape == (4, 4, 4)
    assert out2[1, 1, 1] == data[2, 2, 2]


def test_parcellate_without_transforms(tmp_path):
    """WMH already in atlas space: header-only resampling, on a different (2 mm) grid."""
    atlas = np.zeros((20, 20, 20), np.int16)
    atlas[:10] = 1
    atlas[10:] = 2
    aff = np.eye(4)
    atlas_file = save_labels(atlas, aff, tmp_path / "atlas.nii.gz")
    wmh = np.zeros((10, 10, 10), np.uint8)
    wmh[2, 5, 5] = 1   # x = 4 mm -> region 1
    wmh[7, 5, 5] = 1   # x = 14 mm -> region 2
    aff2 = np.diag([2.0, 2.0, 2.0, 1.0])
    wmh_file = save_labels(wmh, aff2, tmp_path / "wmh.nii.gz", dtype=np.uint8)
    res = parcellate_wmh(wmh_file, str(atlas_file), None, out_dir=tmp_path / "out", prefix="t", split_masks=True,
                         quiet=True)
    df = res.volumes
    assert list(df["wmh_voxels"]) == [1, 1, 0]
    assert np.isclose(df.loc[0, "wmh_volume_ml"], 0.008)
    assert res.volumes_tsv.exists() and res.wmh_parc_image.exists()
    assert len(res.masks) == 2 and all(p.exists() for p in res.masks.values())
    parc = np.asarray(nib.load(str(res.wmh_parc_image)).dataobj)
    assert parc[2, 5, 5] == 1 and parc[7, 5, 5] == 2
