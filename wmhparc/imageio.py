"""Small NIfTI helpers shared across the package (no ANTs dependency)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

PathLike = Union[str, Path]


def load_labels(path: PathLike) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a label image as int32 (values are rounded, so float-stored labels such as
    the CoBrALab 1.0000153 noise are handled).  Returns (data, affine, header)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim == 4:
        if data.shape[3] != 1:
            raise ValueError(f"{path}: expected a 3-D label image, got shape {data.shape}")
        data = data[..., 0]
    data = np.rint(data).astype(np.int32)
    return data, img.affine.copy(), img.header


def load_mask(path: PathLike, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a binary mask (probabilistic maps are thresholded at ``threshold``)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    return (data > threshold), img.affine.copy(), img.header


def save_labels(data: np.ndarray, affine: np.ndarray, path: PathLike, dtype=np.int16) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if dtype is not None:
        data = data.astype(dtype)
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return path


def voxel_volume_ml(affine: np.ndarray) -> float:
    """Voxel volume in millilitres from an affine."""
    return float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0


def voxel_spacing(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def same_grid(aff_a: np.ndarray, shape_a, aff_b: np.ndarray, shape_b, atol: float = 1e-3) -> bool:
    return tuple(shape_a[:3]) == tuple(shape_b[:3]) and np.allclose(aff_a, aff_b, atol=atol)


def resample_to_grid(data: np.ndarray, src_affine: np.ndarray, dst_affine: np.ndarray, dst_shape,
                     order: int = 0, cval: float = 0.0) -> np.ndarray:
    """Resample ``data`` (defined on ``src_affine``) onto the grid (``dst_affine``, ``dst_shape``)
    using only the header affines (no registration).  ``order=0`` = nearest neighbour (labels);
    ``order=1`` = trilinear.  Use this when two images are already in the same physical space
    (e.g. a 2 mm and a 1 mm MNI image, or the 0.5 mm Chung grid and a 1 mm MNI grid)."""
    dst_shape = tuple(int(s) for s in dst_shape[:3])
    # voxel(dst) -> world -> voxel(src)
    m = np.linalg.inv(src_affine) @ dst_affine
    ii, jj, kk = np.meshgrid(np.arange(dst_shape[0]), np.arange(dst_shape[1]), np.arange(dst_shape[2]),
                             indexing="ij")
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], axis=0)
    src = (m @ coords)[:3]
    if order == 0:
        src = np.rint(src)
    out = ndi.map_coordinates(data, src, order=order, mode="constant", cval=cval, prefilter=False)
    return out.reshape(dst_shape).astype(data.dtype if order == 0 else np.float32)


def world_coords(shape, affine: np.ndarray) -> np.ndarray:
    """(X, Y, Z, 3) array of world coordinates for every voxel."""
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape[:3]], indexing="ij")
    vox = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1)
    return (vox @ affine.T)[..., :3]
