"""Bullseye white-matter parcellation (lobe x depth layer).

Implements the scheme of Sudre et al. (2018) as realised in Gerard Sanroma's
``bullseye_pipeline`` (the pipeline Chung et al. used to make their 20-ROI atlas):

1. *Depth*: normalised distance ``d = D_vent / (D_vent + D_cortex)`` where ``D_vent`` is the
   Euclidean distance to the lateral ventricles and ``D_cortex`` the distance to the (lobar)
   cortex.  Four equal-width layers of ``d`` give layer 1 (periventricular, ``d < 0.25``) to
   layer 4 (juxtacortical, ``d >= 0.75``).
2. *Lobes*: cortical lobe labels (frontal, parietal, temporal, occipital; insula counted with
   frontal, cingulate left unlabelled) are propagated into the white matter of the same
   hemisphere by nearest-cortex assignment (what ``mri_aparc2aseg --labelwm`` does); unlabelled
   WM (e.g. under the cingulate) is filled from the nearest labelled voxel.
3. *BGIT*: basal ganglia + thalamus override the lobar labels as a ninth region.
4. Region x layer gives 36 labels (9 regions x 4 layers).  Folding hemispheres and ordering
   the regions BGIT, frontal, occipital, temporal, parietal gives the 20-label numbering used
   by Chung et al.

Everything here is pure numpy/scipy and runs on any label image (FreeSurfer ``aparc+aseg``,
SynthSeg + a lobe atlas, or the Harvard-Oxford atlases in MNI space).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import importlib.resources as ir
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .imageio import load_labels, save_labels, voxel_spacing

# ----------------------------------------------------------------------------------------------
# Region coding used internally.  Lobe codes are hemisphere*10 + lobe.
# ----------------------------------------------------------------------------------------------
LOBES = ("frontal", "parietal", "temporal", "occipital")
LOBE_ID = {"frontal": 1, "parietal": 2, "temporal": 3, "occipital": 4}
LEFT, RIGHT = 1, 2
BGIT_CODE = 5


def lobe_code(lobe: str, hemisphere: int) -> int:
    return hemisphere * 10 + LOBE_ID[lobe]


# bullseye36 region order (matches data/luts/bullseye36.tsv and the Edinburgh tool)
REGIONS_36: Tuple[int, ...] = (
    lobe_code("frontal", LEFT), lobe_code("parietal", LEFT), lobe_code("temporal", LEFT), lobe_code("occipital", LEFT),
    BGIT_CODE,
    lobe_code("frontal", RIGHT), lobe_code("parietal", RIGHT), lobe_code("temporal", RIGHT), lobe_code("occipital", RIGHT),
)
# chung20 / bullseye20 region order (hemispheres folded)
REGIONS_20: Tuple[str, ...] = ("BGIT", "frontal", "occipital", "temporal", "parietal")

N_LAYERS = 4


# ----------------------------------------------------------------------------------------------
# Core geometry
# ----------------------------------------------------------------------------------------------
def normalized_depth(ventricles: np.ndarray, cortex: np.ndarray, spacing: Sequence[float] = (1, 1, 1)) -> np.ndarray:
    """Normalised distance from the ventricles (0) to the cortex (1), computed everywhere."""
    ventricles = np.asarray(ventricles, bool)
    cortex = np.asarray(cortex, bool)
    if not ventricles.any() or not cortex.any():
        raise ValueError("ventricle and cortex masks must both be non-empty")
    d_v = ndi.distance_transform_edt(~ventricles, sampling=spacing)
    d_c = ndi.distance_transform_edt(~cortex, sampling=spacing)
    denom = d_v + d_c
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = np.where(denom > 0, d_v / denom, 0.0)
    return nd.astype(np.float32)


def depth_layers(ndist: np.ndarray, mask: np.ndarray, n_layers: int = N_LAYERS) -> np.ndarray:
    """Bin the normalised depth into ``n_layers`` equal-width layers (1 = innermost) within ``mask``.

    Voxels with ``ndist == 0`` (inside the ventricles) get 0; ``ndist == 1`` (inside the cortex)
    gets the outermost layer so that cortex-adjacent WM never falls out of the parcellation.
    """
    layers = np.zeros(ndist.shape, np.int8)
    edges = np.linspace(0.0, 1.0, n_layers + 1)
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        sel = (ndist >= lo) & ((ndist < hi) if i < n_layers - 1 else (ndist <= hi))
        layers[sel] = i + 1
    layers[np.isclose(ndist, 0.0)] = 0
    layers[~np.asarray(mask, bool)] = 0
    return layers


def propagate_labels(seeds: np.ndarray, target: np.ndarray, hemisphere: Optional[np.ndarray] = None,
                     spacing: Sequence[float] = (1, 1, 1), seed_hemisphere: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """Give every ``target`` voxel the label of the nearest non-zero ``seeds`` voxel.

    If ``hemisphere`` (0 = unknown, 1 = left, 2 = right, defined on the target voxels) is given,
    voxels only take seeds from their own hemisphere; voxels of unknown hemisphere use all seeds.
    The hemisphere of each seed is read from ``seed_hemisphere`` (defaults to ``hemisphere``).
    """
    seeds = np.asarray(seeds)
    target = np.asarray(target, bool)
    out = np.zeros(seeds.shape, seeds.dtype)
    groups: Iterable[Tuple[np.ndarray, np.ndarray]]
    if hemisphere is None:
        groups = [(seeds != 0, target)]
    else:
        hemisphere = np.asarray(hemisphere)
        sh = hemisphere if seed_hemisphere is None else np.asarray(seed_hemisphere)
        groups = [((seeds != 0) & (sh == LEFT), target & (hemisphere == LEFT)),
                  ((seeds != 0) & (sh == RIGHT), target & (hemisphere == RIGHT)),
                  ((seeds != 0), target & (hemisphere == 0))]
    for seed_mask, tgt in groups:
        if not tgt.any():
            continue
        if not seed_mask.any():
            raise ValueError("no seeds available for a hemisphere that has target voxels")
        _, idx = ndi.distance_transform_edt(~seed_mask, sampling=spacing, return_distances=True, return_indices=True)
        nearest = seeds[idx[0][tgt], idx[1][tgt], idx[2][tgt]]
        out[tgt] = nearest
    return out


# ----------------------------------------------------------------------------------------------
# Generic builder
# ----------------------------------------------------------------------------------------------
@dataclass
class BullseyeInputs:
    """Tissue inputs on a common grid.

    ventricles : bool  lateral ventricles (distance origin)
    cortex     : bool  cortical target for the depth (lobar cortex; by default without insula/cingulate)
    wm         : bool  cerebral white matter to parcellate
    bgt        : bool  basal ganglia + thalamus (becomes the BGIT region; parcellated by depth as well)
    lobe_seeds : int   lobe codes (``lobe_code``) on cortical voxels used to label the WM
    hemisphere : int   1 = left, 2 = right, 0 = unknown, for WM voxels
    """
    ventricles: np.ndarray
    cortex: np.ndarray
    wm: np.ndarray
    bgt: np.ndarray
    lobe_seeds: np.ndarray
    hemisphere: np.ndarray
    affine: np.ndarray

    @property
    def spacing(self) -> np.ndarray:
        return voxel_spacing(self.affine)


@dataclass
class BullseyeResult:
    labels36: np.ndarray
    labels20: np.ndarray
    layers: np.ndarray
    lobes: np.ndarray       # region codes (lobe codes + BGIT_CODE)
    ndist: np.ndarray
    affine: np.ndarray

    def save(self, prefix: Path, quiet: bool = False) -> Dict[str, Path]:
        prefix = Path(prefix)
        out = {
            "bullseye36": save_labels(self.labels36, self.affine, f"{prefix}_atlas-bullseye36_dseg.nii.gz"),
            "bullseye20": save_labels(self.labels20, self.affine, f"{prefix}_atlas-bullseye20_dseg.nii.gz"),
            "layers": save_labels(self.layers, self.affine, f"{prefix}_desc-layers_dseg.nii.gz"),
            "lobes": save_labels(self.lobes, self.affine, f"{prefix}_desc-lobes_dseg.nii.gz"),
            "ndist": save_labels(self.ndist, self.affine, f"{prefix}_desc-ndist.nii.gz", dtype=np.float32),
        }
        if not quiet:
            for k, p in out.items():
                print(f"[wmhparc] wrote {k}: {p}")
        return out


def combine_regions_layers(regions: np.ndarray, layers: np.ndarray, region_order: Sequence[int],
                           n_layers: int = N_LAYERS) -> np.ndarray:
    """label = position(region in region_order) * n_layers + layer."""
    out = np.zeros(regions.shape, np.int16)
    for gi, code in enumerate(region_order):
        for layer in range(1, n_layers + 1):
            out[(regions == code) & (layers == layer)] = gi * n_layers + layer
    return out


def fold_hemispheres(labels36: np.ndarray, n_layers: int = N_LAYERS) -> np.ndarray:
    """bullseye36 -> bullseye20 (chung20 numbering: BGIT, frontal, occipital, temporal, parietal)."""
    lut36 = _lut("bullseye36")
    out = np.zeros(labels36.shape, np.int16)
    for _, row in lut36.iterrows():
        gi = REGIONS_20.index(row["region"])
        out[labels36 == row["index"]] = gi * n_layers + int(row["layer"])
    return out


def build_bullseye(inp: BullseyeInputs, n_layers: int = N_LAYERS) -> BullseyeResult:
    spacing = inp.spacing
    ndist = normalized_depth(inp.ventricles, inp.cortex, spacing)
    parc_mask = (inp.wm | inp.bgt) & ~inp.ventricles
    layers = depth_layers(ndist, parc_mask, n_layers)
    # lobes: nearest cortical seed of the same hemisphere, then BGIT override
    # lobe codes carry their hemisphere (code // 10), so seeds need no hemisphere image
    lobes = propagate_labels(inp.lobe_seeds, inp.wm & ~inp.bgt, inp.hemisphere, spacing,
                             seed_hemisphere=np.asarray(inp.lobe_seeds) // 10)
    lobes = lobes.astype(np.int16)
    lobes[inp.bgt] = BGIT_CODE
    lobes[~parc_mask] = 0
    labels36 = combine_regions_layers(lobes, layers, REGIONS_36, n_layers)
    labels20 = fold_hemispheres(labels36, n_layers)
    return BullseyeResult(labels36, labels20, layers, lobes, ndist, inp.affine)


# ----------------------------------------------------------------------------------------------
# Adapters: FreeSurfer aparc+aseg
# ----------------------------------------------------------------------------------------------
# Desikan-Killiany cortical parcels (left = 1000 + k, right = 2000 + k) -> lobe, after FreeSurfer's
# ``mri_annotation2label --lobesStrict``.
DESIKAN_LOBE: Dict[int, str] = {
    1: "temporal",   # bankssts
    2: "cingulate",  # caudalanteriorcingulate
    3: "frontal",    # caudalmiddlefrontal
    5: "occipital",  # cuneus
    6: "temporal",   # entorhinal
    7: "temporal",   # fusiform
    8: "parietal",   # inferiorparietal
    9: "temporal",   # inferiortemporal
    10: "cingulate",  # isthmuscingulate
    11: "occipital",  # lateraloccipital
    12: "frontal",   # lateralorbitofrontal
    13: "occipital",  # lingual
    14: "frontal",   # medialorbitofrontal
    15: "temporal",  # middletemporal
    16: "temporal",  # parahippocampal
    17: "frontal",   # paracentral
    18: "frontal",   # parsopercularis
    19: "frontal",   # parsorbitalis
    20: "frontal",   # parstriangularis
    21: "occipital",  # pericalcarine
    22: "parietal",  # postcentral
    23: "cingulate",  # posteriorcingulate
    24: "frontal",   # precentral
    25: "parietal",  # precuneus
    26: "cingulate",  # rostralanteriorcingulate
    27: "frontal",   # rostralmiddlefrontal
    28: "frontal",   # superiorfrontal
    29: "parietal",  # superiorparietal
    30: "temporal",  # superiortemporal
    31: "parietal",  # supramarginal
    32: "frontal",   # frontalpole
    33: "temporal",  # temporalpole
    34: "temporal",  # transversetemporal
    35: "insula",    # insula
}
ASEG_WM = {2: LEFT, 41: RIGHT}
ASEG_WM_HYPO = {77}                  # WM hypointensities (bilateral) -> treated as WM
ASEG_UNSEG_WM = {5001: LEFT, 5002: RIGHT}  # wmparc "UnsegmentedWhiteMatter"
ASEG_LATVENT = {4: LEFT, 43: RIGHT}
ASEG_INFLATVENT = {5: LEFT, 44: RIGHT}
ASEG_BGT = {10: LEFT, 11: LEFT, 12: LEFT, 13: LEFT, 26: LEFT,      # thalamus, caudate, putamen, pallidum, accumbens
            49: RIGHT, 50: RIGHT, 51: RIGHT, 52: RIGHT, 58: RIGHT}


def _resolve_lobe(lobe: str, insula_to: str) -> Optional[str]:
    """Map a raw lobe name to a seed lobe or None (not a seed)."""
    if lobe in LOBES:
        return lobe
    if lobe == "insula":
        return None if insula_to == "none" else insula_to
    return None  # cingulate and anything else: filled from neighbours


def inputs_from_freesurfer(aparc_aseg: np.ndarray, affine: np.ndarray, insula_to: str = "frontal",
                           include_inferior_lateral_ventricles: bool = False,
                           cortex_target_excludes: Sequence[str] = ("insula", "cingulate")) -> BullseyeInputs:
    """Build :class:`BullseyeInputs` from a FreeSurfer ``aparc+aseg`` (or ``wmparc``) label volume.

    ``insula_to`` : ``"frontal"`` (Sanroma/Chung convention), ``"bgit"`` or ``"none"``.
    """
    seg = np.asarray(aparc_aseg).astype(np.int32)
    ventricles = np.isin(seg, list(ASEG_LATVENT))
    if include_inferior_lateral_ventricles:
        ventricles |= np.isin(seg, list(ASEG_INFLATVENT))
    bgt = np.isin(seg, list(ASEG_BGT))
    wm = np.isin(seg, list(ASEG_WM) + list(ASEG_WM_HYPO) + list(ASEG_UNSEG_WM))
    # wmparc files carry lobe-coded WM (3000+/4000+) -> count as WM of that hemisphere
    wm |= (seg >= 3000) & (seg < 3100)
    wm |= (seg >= 4000) & (seg < 4100)
    hemisphere = np.zeros(seg.shape, np.int8)
    for lab, h in {**ASEG_WM, **ASEG_UNSEG_WM, **ASEG_BGT}.items():
        hemisphere[seg == lab] = h
    hemisphere[(seg >= 3000) & (seg < 3100)] = LEFT
    hemisphere[(seg >= 4000) & (seg < 4100)] = RIGHT
    seeds = np.zeros(seg.shape, np.int16)
    cortex = np.zeros(seg.shape, bool)
    for base, h in ((1000, LEFT), (2000, RIGHT)):
        for k, lobe in DESIKAN_LOBE.items():
            m = seg == base + k
            if not m.any():
                continue
            if lobe not in cortex_target_excludes:
                cortex |= m
            tgt = _resolve_lobe(lobe, insula_to)
            if tgt == "bgit":
                bgt |= m
            elif tgt is not None:
                seeds[m] = lobe_code(tgt, h)
    # WM hypointensities (77) have no hemisphere: take the nearest labelled hemisphere
    unknown = wm & (hemisphere == 0)
    if unknown.any():
        hemisphere[unknown] = propagate_labels(hemisphere, unknown, None, voxel_spacing(affine))[unknown]
    return BullseyeInputs(ventricles, cortex, wm, bgt, seeds, hemisphere, affine)


def build_from_freesurfer(aparc_aseg_path, out_prefix, quiet: bool = False, **kw) -> BullseyeResult:
    """Subject-space bullseye from a FreeSurfer/FastSurfer ``aparc+aseg`` NIfTI (convert .mgz first with
    ``mri_convert``)."""
    seg, affine, _ = load_labels(aparc_aseg_path)
    res = build_bullseye(inputs_from_freesurfer(seg, affine, **kw))
    res.save(Path(out_prefix), quiet=quiet)
    return res


# ----------------------------------------------------------------------------------------------
# Adapters: SynthSeg (aseg-style labels, no cortical parcellation) + a lobe label image
# ----------------------------------------------------------------------------------------------
def inputs_from_synthseg(synthseg: np.ndarray, lobes: np.ndarray, affine: np.ndarray,
                         include_inferior_lateral_ventricles: bool = True) -> BullseyeInputs:
    """SynthSeg/aseg tissue labels + a lobe image (codes from :func:`lobe_code`, e.g. a warped atlas).

    ``lobes`` gives the lobe for cortical (and optionally WM) voxels; cortex voxels whose lobe is
    0 are not used as depth targets.  Cortex = aseg 3/42, WM = 2/41, ventricles 4/43 (+5/44).
    """
    seg = np.asarray(synthseg).astype(np.int32)
    lobes = np.asarray(lobes).astype(np.int16)
    ventricles = np.isin(seg, list(ASEG_LATVENT))
    if include_inferior_lateral_ventricles:
        ventricles |= np.isin(seg, list(ASEG_INFLATVENT))
    cortex_all = np.isin(seg, [3, 42])
    wm = np.isin(seg, list(ASEG_WM))
    bgt = np.isin(seg, list(ASEG_BGT))
    hemisphere = np.zeros(seg.shape, np.int8)
    for lab, h in {**ASEG_WM, **ASEG_BGT, 3: LEFT, 42: RIGHT}.items():
        hemisphere[seg == lab] = h
    seeds = np.where(cortex_all & (lobes > 0), lobes, 0).astype(np.int16)
    # make the seed hemisphere consistent with the aseg hemisphere
    seeds = np.where(seeds > 0, (hemisphere * 10 + seeds % 10), 0).astype(np.int16)
    cortex = cortex_all & (seeds > 0)
    return BullseyeInputs(ventricles, cortex, wm, bgt, seeds, hemisphere, affine)


# ----------------------------------------------------------------------------------------------
# Adapters: Harvard-Oxford atlases in MNI152NLin6Asym (TemplateFlow)
# ----------------------------------------------------------------------------------------------
HOSPA = {  # TemplateFlow HOSPA th0 dseg values (== Harvard-Oxford subcortical index)
    "wm": {1: LEFT, 12: RIGHT}, "cortex": {2: LEFT, 13: RIGHT}, "latvent": {3: LEFT, 14: RIGHT},
    "bgt": {4: LEFT, 5: LEFT, 6: LEFT, 7: LEFT, 11: LEFT, 15: RIGHT, 16: RIGHT, 17: RIGHT, 18: RIGHT, 21: RIGHT},
}


def harvard_oxford_lobe_table() -> pd.DataFrame:
    with ir.as_file(ir.files("wmhparc").joinpath("data", "harvard_oxford_cortical_lobes.tsv")) as p:
        return pd.read_csv(p, sep="\t")


def inputs_from_harvard_oxford(hospa: np.ndarray, hocpal: np.ndarray, affine: np.ndarray,
                               insula_to: str = "frontal",
                               cortex_target_excludes: Sequence[str] = ("insula", "cingulate")) -> BullseyeInputs:
    hospa = np.asarray(hospa).astype(np.int32)
    hocpal = np.asarray(hocpal).astype(np.int32)
    tbl = harvard_oxford_lobe_table()
    ventricles = np.isin(hospa, list(HOSPA["latvent"]))
    wm = np.isin(hospa, list(HOSPA["wm"]))
    bgt = np.isin(hospa, list(HOSPA["bgt"]))
    hemisphere = np.zeros(hospa.shape, np.int8)
    for grp in ("wm", "cortex", "bgt", "latvent"):
        for lab, h in HOSPA[grp].items():
            hemisphere[hospa == lab] = h
    seeds = np.zeros(hospa.shape, np.int16)
    cortex = np.zeros(hospa.shape, bool)
    for _, row in tbl.iterrows():
        m = hocpal == int(row["value"])
        if not m.any():
            continue
        h = LEFT if row["hemisphere"] == "left" else RIGHT
        lobe = str(row["lobe"])
        if lobe not in cortex_target_excludes:
            cortex |= m
        tgt = _resolve_lobe(lobe, insula_to)
        if tgt == "bgit":
            bgt |= m
        elif tgt is not None:
            seeds[m] = lobe_code(tgt, h)
    # cortex voxels of the subcortical atlas without a cortical-atlas label still count as cortex
    cortex |= np.isin(hospa, list(HOSPA["cortex"])) & (hocpal == 0)
    cortex &= ~wm & ~bgt & ~ventricles
    return BullseyeInputs(ventricles, cortex, wm, bgt, seeds, hemisphere, affine)


def build_mni_bullseye(out_prefix: Optional[Path] = None, quiet: bool = False, **kw) -> BullseyeResult:
    """Build the 36/20-region bullseye in MNI152NLin6Asym from the Harvard-Oxford atlases."""
    from .templates import get_template
    tpl = get_template("MNI152NLin6Asym")
    hospa, aff, _ = load_labels(tpl.fetch("HOSPA", quiet=quiet))
    hocpal, aff2, _ = load_labels(tpl.fetch("HOCPAL", quiet=quiet))
    if hospa.shape != hocpal.shape or not np.allclose(aff, aff2):
        raise RuntimeError("HOSPA and HOCPAL atlases are not on the same grid")
    res = build_bullseye(inputs_from_harvard_oxford(hospa, hocpal, aff, **kw))
    if out_prefix is not None:
        res.save(Path(out_prefix), quiet=quiet)
    return res


def _mni_prefix() -> Path:
    from .atlases import derived_dir
    return derived_dir() / "mni_bullseye"


def build_mni_bullseye36(out: Path, quiet: bool = False) -> Path:
    """Builder hook used by :func:`wmhparc.atlases.atlas_path` for ``bullseye36``."""
    res = build_mni_bullseye(_mni_prefix(), quiet=quiet)
    save_labels(res.labels36, res.affine, out)
    from .atlases import derived_dir
    save_labels(res.labels20, res.affine, derived_dir() / "bullseye20_space-MNI152NLin6Asym_dseg.nii.gz")
    return Path(out)


def build_mni_bullseye20(out: Path, quiet: bool = False) -> Path:
    res = build_mni_bullseye(_mni_prefix(), quiet=quiet)
    save_labels(res.labels20, res.affine, out)
    from .atlases import derived_dir
    save_labels(res.labels36, res.affine, derived_dir() / "bullseye36_space-MNI152NLin6Asym_dseg.nii.gz")
    return Path(out)


def _lut(name: str) -> pd.DataFrame:
    from .atlases import load_lut
    return load_lut(name)
