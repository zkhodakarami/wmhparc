"""Apply a standard-space parcellation to a WMH mask and compute regional volumes.

Two modes:

* ``space="native"`` (recommended for volumes): the atlas is brought onto the WMH grid with
  nearest-neighbour / label interpolation through the subject's transform chain and
  intersected with the WMH mask.  Volumes are real subject volumes.
* ``space="template"``: the WMH mask is warped to the template grid and intersected with the
  atlas there.  Handy for voxel-wise group analyses; volumes are template-space volumes.

If no transform chain is given the WMH mask is assumed to be in the atlas' MNI space
already and only header-based resampling is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .atlases import atlas_path, load_lut, ATLASES
from .imageio import load_labels, load_mask, resample_to_grid, same_grid, save_labels, voxel_spacing, voxel_volume_ml
from .registration import TransformChain, apply_transforms
from .templates import get_template

PathLike = Union[str, Path]


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------
def regional_volumes(wmh: np.ndarray, atlas: np.ndarray, affine: np.ndarray, lut: pd.DataFrame,
                     include_unlabelled: bool = True) -> pd.DataFrame:
    """Per-region WMH voxel counts / volumes for a WMH mask and a label image on the same grid.

    Columns: index, name (+ any extra LUT columns), wmh_voxels, wmh_volume_ml, region_volume_ml,
    wmh_fraction_of_region (WMH load of the region), fraction_of_total_wmh (distribution of the
    subject's WMH over regions).  A final row ``index=0`` counts WMH voxels outside the atlas.
    """
    wmh = np.asarray(wmh, bool)
    atlas = np.asarray(atlas)
    if wmh.shape != atlas.shape:
        raise ValueError(f"shape mismatch: wmh {wmh.shape} vs atlas {atlas.shape}")
    vv = voxel_volume_ml(affine)
    total = int(wmh.sum())
    rows = []
    for _, r in lut.iterrows():
        idx = int(r["index"])
        reg = atlas == idx
        n = int((reg & wmh).sum())
        nreg = int(reg.sum())
        row = dict(r)
        row.update(wmh_voxels=n, wmh_volume_ml=n * vv, region_volume_ml=nreg * vv,
                   wmh_fraction_of_region=(n / nreg if nreg else np.nan),
                   fraction_of_total_wmh=(n / total if total else np.nan))
        rows.append(row)
    if include_unlabelled:
        n = int((wmh & (atlas == 0)).sum())
        row = {c: ("unlabelled" if c == "name" else (0 if c == "index" else None)) for c in lut.columns}
        row.update(wmh_voxels=n, wmh_volume_ml=n * vv, region_volume_ml=np.nan, wmh_fraction_of_region=np.nan,
                   fraction_of_total_wmh=(n / total if total else np.nan))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["total_wmh_voxels"] = total
    df.attrs["total_wmh_volume_ml"] = total * vv
    return df


def fill_unlabelled(wmh: np.ndarray, atlas: np.ndarray, affine: np.ndarray, max_mm: float = 10.0) -> np.ndarray:
    """Return a copy of ``atlas`` where WMH voxels with label 0 take the nearest label within ``max_mm``.

    Useful for parcellations that do not cover the whole white matter (the CoBrALab clusters
    cover only voxels where WMH are prevalent; the Chung atlas covers cerebral WM only).
    """
    atlas = np.asarray(atlas).copy()
    todo = np.asarray(wmh, bool) & (atlas == 0)
    if not todo.any():
        return atlas
    dist, idx = ndi.distance_transform_edt(atlas == 0, sampling=voxel_spacing(affine), return_indices=True)
    ok = todo & (dist <= max_mm)
    atlas[ok] = atlas[idx[0][ok], idx[1][ok], idx[2][ok]]
    return atlas


# ----------------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------------
@dataclass
class ParcellationResult:
    atlas_name: str
    space: str
    atlas_image: Path          # atlas resampled to the output grid
    wmh_parc_image: Path       # WMH voxels carrying their region label
    volumes_tsv: Path
    volumes: pd.DataFrame
    masks: Dict[int, Path]


def _atlas_source(atlas: str, template: Optional[str]) -> tuple[Path, pd.DataFrame, str]:
    """(label image path, LUT, name) for an atlas name or an arbitrary label file."""
    if atlas in ATLASES:
        return atlas_path(atlas, space=template), load_lut(atlas), atlas
    p = Path(atlas)
    if not p.exists():
        raise FileNotFoundError(f"{atlas!r} is neither a known atlas ({sorted(ATLASES)}) nor a file")
    data, _, _ = load_labels(p)
    labs = [int(v) for v in np.unique(data) if v != 0]
    lut = pd.DataFrame({"index": labs, "name": [f"label{v}" for v in labs]})
    return p, lut, p.name.replace(".nii.gz", "").replace(".nii", "")


def parcellate_wmh(wmh: PathLike, atlas: str, chain: Optional[Union[TransformChain, PathLike]] = None,
                   out_dir: PathLike = "wmhparc_out", prefix: str = "sub", space: str = "native",
                   wmh_modality: str = "FLAIR", split_masks: bool = False, fill_mm: Optional[float] = None,
                   lut: Optional[pd.DataFrame] = None, backend: str = "auto", quiet: bool = False,
                   ) -> ParcellationResult:
    """Parcellate a binary WMH mask with a standard-space atlas.

    Parameters
    ----------
    wmh : path to a binary WMH mask (FLAIR space by default; see ``wmh_modality``).
    atlas : registry name (``chung20``, ``cobralab_patho``, ``bullseye36``, ...) or a label NIfTI.
    chain : :class:`TransformChain` or its ``transforms.json`` from :func:`register_subject`.
            ``None`` means the WMH mask is already in the atlas' MNI space.
    space : ``"native"`` (atlas -> WMH grid) or ``"template"`` (WMH -> template grid).
    wmh_modality : ``"FLAIR"`` or ``"T1w"``: which subject image the WMH mask is aligned with.
    split_masks : also write one binary NIfTI per region.
    fill_mm : assign unlabelled WMH voxels to the nearest region within this distance (mm).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if chain is not None and not isinstance(chain, TransformChain):
        chain = TransformChain.load(chain)
    template = chain.template if chain is not None else None
    atlas_file, lut_df, name = _atlas_source(atlas, template)
    if lut is not None:
        lut_df = lut
    wmh_path = Path(wmh)
    tag = f"{prefix}_atlas-{name}_space-{space}"

    if space == "native":
        wmh_data, affine, _ = load_mask(wmh_path)
        atlas_out = out_dir / f"{tag}_dseg.nii.gz"
        if chain is None:
            a_data, a_aff, _ = load_labels(atlas_file)
            if same_grid(a_aff, a_data.shape, affine, wmh_data.shape):
                atlas_native = a_data
            else:
                atlas_native = resample_to_grid(a_data, a_aff, affine, wmh_data.shape, order=0)
            save_labels(atlas_native, affine, atlas_out)
        else:
            xf, inv = chain.from_template(chain.space_of(wmh_modality))
            apply_transforms(wmh_path, atlas_file, atlas_out, xf, inv, interpolation="label", backend=backend)
            atlas_native, _, _ = load_labels(atlas_out)
        atlas_grid, grid_aff = atlas_native, affine
        wmh_grid = wmh_data
    elif space == "template":
        if chain is None:
            raise ValueError("space='template' needs a transform chain (otherwise the mask is already in template space)")
        tpl = get_template(chain.template)
        ref = tpl.fetch("T1w", quiet=True)
        wmh_out = out_dir / f"{tag}_desc-wmh_mask.nii.gz"
        xf, inv = chain.to_template(chain.space_of(wmh_modality))
        apply_transforms(ref, wmh_path, wmh_out, xf, inv, interpolation="label", backend=backend)
        wmh_grid, grid_aff, _ = load_mask(wmh_out)
        a_data, a_aff, _ = load_labels(atlas_file)
        if same_grid(a_aff, a_data.shape, grid_aff, wmh_grid.shape):
            atlas_grid = a_data
        else:
            atlas_grid = resample_to_grid(a_data, a_aff, grid_aff, wmh_grid.shape, order=0)
        atlas_out = save_labels(atlas_grid, grid_aff, out_dir / f"{tag}_dseg.nii.gz")
    else:
        raise ValueError("space must be 'native' or 'template'")

    if fill_mm:
        atlas_grid = fill_unlabelled(wmh_grid, atlas_grid, grid_aff, max_mm=fill_mm)
        atlas_out = save_labels(atlas_grid, grid_aff, out_dir / f"{tag}_desc-filled_dseg.nii.gz")

    wmh_parc = np.where(wmh_grid, atlas_grid, 0).astype(np.int16)
    parc_out = save_labels(wmh_parc, grid_aff, out_dir / f"{tag}_desc-wmhparc_dseg.nii.gz")
    vols = regional_volumes(wmh_grid, atlas_grid, grid_aff, lut_df)
    vols.insert(0, "subject", prefix)
    vols.insert(1, "atlas", name)
    vols.insert(2, "space", space)
    tsv = out_dir / f"{tag}_volumes.tsv"
    vols.to_csv(tsv, sep="\t", index=False, float_format="%.6g")

    masks: Dict[int, Path] = {}
    if split_masks:
        mdir = out_dir / f"{tag}_masks"
        for _, r in lut_df.iterrows():
            idx = int(r["index"])
            m = wmh_parc == idx
            safe = str(r["name"]).replace(" ", "-").replace("/", "-")
            masks[idx] = save_labels(m.astype(np.uint8), grid_aff, mdir / f"{tag}_label-{idx:02d}_desc-{safe}_mask.nii.gz",
                                     dtype=np.uint8)
    if not quiet:
        print(f"[wmhparc] {name} ({space}): total WMH {vols.attrs['total_wmh_volume_ml']:.3f} ml, "
              f"unlabelled {vols.loc[vols['index'] == 0, 'wmh_volume_ml'].sum():.3f} ml -> {tsv}")
    return ParcellationResult(name, space, Path(atlas_out), parc_out, tsv, vols, masks)


def parcellate_many(wmh: PathLike, atlases: Sequence[str], chain, out_dir: PathLike, prefix: str = "sub",
                    **kw) -> List[ParcellationResult]:
    results = [parcellate_wmh(wmh, a, chain, out_dir=out_dir, prefix=prefix, **kw) for a in atlases]
    long = pd.concat([r.volumes for r in results], ignore_index=True)
    long.to_csv(Path(out_dir) / f"{prefix}_desc-allatlases_volumes.tsv", sep="\t", index=False, float_format="%.6g")
    return results
