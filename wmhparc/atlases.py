"""Registry of WMH parcellations in standard space.

Released parcellations are downloaded on first use (SHA-256 pinned); the
bullseye parcellations are built locally from TemplateFlow resources and cached.

Label tables are shipped as TSV files in ``wmhparc/data/luts``.  Notes on
provenance of each table are in the README and in each entry below.
"""
from __future__ import annotations

import importlib.resources as ir
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .cache import cache_dir
from .templates import Resource

CHUNG_URL = "https://raw.githubusercontent.com/JinyongChung/WMH_progression_modeling/main"
COBRA_URL = "https://raw.githubusercontent.com/CoBrALab/WMH_patho_UKB/main/data"


@dataclass(frozen=True)
class Atlas:
    """A labelled parcellation image plus metadata."""
    name: str
    description: str
    space: str                       # coordinate space the label image is defined in
    lut: str                         # LUT name under data/luts
    resources: Dict[str, Resource] = field(default_factory=dict)
    label_key: str = "labels"        # which resource is the label image
    builder: Optional[str] = None    # name of a function in wmhparc.bullseye that builds it (built atlases)
    mni_aligned: bool = True         # False => needs align_template() before use in MNI
    citation: str = ""
    url: str = ""
    notes: str = ""

    @property
    def is_built(self) -> bool:
        return self.builder is not None


def _res(url: str, relpath: str, sha: str) -> Resource:
    return Resource(url, sha, relpath)


ATLASES: Dict[str, Atlas] = {
    "chung20": Atlas(
        name="chung20",
        description="Chung et al. 20-ROI bullseye-style WM parcellation (5 regions x 4 depth layers), 0.5 mm",
        space="Colin27 (ch2better) 0.5 mm grid, MNI152-aligned",
        lut="chung20",
        resources={"labels": _res(f"{CHUNG_URL}/MNI_ch2better_WM_20ROIs.nii.gz",
                                  "atlases/chung20/MNI_ch2better_WM_20ROIs.nii.gz",
                                  "252da6297879352fc2cdc16e9e902296cdecadeb1c32f468998a67a6f9bc27a7")},
        citation="Chung J. et al. Distinct spatiotemporal patterns of white matter hyperintensity progression. "
                 "Nat Commun 2025.",
        url="https://github.com/JinyongChung/WMH_progression_modeling",
        notes=("The repository ships no label table. The table in data/luts/chung20.tsv was read off the file: "
               "labels form five chains of four mutually adjacent ROIs (1-4, 5-8, 9-12, 13-16, 17-20); within each "
               "chain the first label is smallest, hugs the lateral ventricles and touches the first label of the "
               "other chains (periventricular), the fourth is largest and reaches the cortex (juxtacortical). "
               "Region identity from bounding boxes in MNI mm: 1-4 central (basal ganglia/thalamus, BGIT), 5-8 "
               "anterior (frontal), 9-12 posterior (occipital), 13-16 inferior-lateral (temporal), 17-20 "
               "superior-posterior (parietal). No hemisphere split. Grid: 301x370x316 at 0.5 mm, origin "
               "(-75,-107,-69.5); Colin27 is affinely aligned to MNI152, so treat as MNI152 for lobe/layer work."),
    ),
    "cobralab_patho": Atlas(
        name="cobralab_patho",
        description="CoBrALab data-driven WMH pathophysiology parcellation (periventricular / posterior / anterior)",
        space="MNI152NLin2009cAsym",
        lut="cobralab_patho",
        resources={"labels": _res(f"{COBRA_URL}/WMH_parc_patho_MNI.nii.gz",
                                  "atlases/cobralab/WMH_parc_patho_MNI.nii.gz",
                                  "179437d8612b0a0e24c506cee31b74eb0435bf9ba9f02806993fe9d23b533ece")},
        citation="Parent O. et al. Characterizing spatiotemporal white matter hyperintensity pathophysiology in vivo "
                 "to disentangle vascular and neurodegenerative contributions. Nat Commun 2026.",
        url="https://github.com/CoBrALab/WMH_patho_UKB",
        notes="Label meanings taken from analyses/2_spatial_clust/2e_overlap_other_parc.R in the release "
              "(Periventricular=1, Posterior=2, Anterior=3). Grid 193x229x193 1 mm == TemplateFlow MNI152NLin2009cAsym res-01.",
    ),
}

# CoBrALab comparison parcellations: released in their custom UK Biobank template space only.
_COBRA_UKB = {
    "cobralab_lobar": ("WMH_parc_lobar.nii.gz", "d6127159df9aa61600e42bf977ab4149235b73ed71e5ae6de9fdf7d6f08a03f9",
                       "Lobar parcellation of WMH voxels (frontal/temporal/parietal/occipital/cerebellum/brainstem)"),
    "cobralab_pvdeep": ("WMH_parc_pv_deep.nii.gz", "745aa61461a17243d28092383a2943cb0e3855733adbafb11c53d182ff7271f3",
                        "Periventricular vs deep WMH parcellation"),
    "cobralab_vascular": ("WMH_parc_vascular.nii.gz", "2df493e7ecc37c13ea0dcebf1635c67800b47229c315237f0a7605fd150bbca0",
                          "Arterial territory parcellation (ACA/MCA/PCA/VB; from Liu et al. 2023 Sci Data)"),
    "cobralab_fiber": ("WMH_parc_fiber.nii.gz", "f780d5cd84f869244d1bb6dc5ae8de7a0e44040021a7273db3eee84a6f7960df",
                       "Fibre-type parcellation (association/commissural/projection)"),
}
UKB_TEMPLATE = {
    "T1w": _res(f"{COBRA_URL}/UKB_template_T1.nii.gz", "atlases/cobralab/UKB_template_T1.nii.gz",
                "fd95b78be69e72c8b77bc76c7b5377dda899d75090d2e69f6c776cc2a986c843"),
    "mask": _res(f"{COBRA_URL}/UKB_template_mask.nii.gz", "atlases/cobralab/UKB_template_mask.nii.gz",
                 "310366fce5c30dc8188456f0877d342f34caadbb594eb88e28a6e95334c41ffd"),
}
for _name, (_fname, _sha, _desc) in _COBRA_UKB.items():
    ATLASES[_name] = Atlas(
        name=_name, description=f"CoBrALab {_desc}", space="UKB (CoBrALab custom UK Biobank T1 template, 2 mm)",
        lut=_name,
        resources={"labels": _res(f"{COBRA_URL}/{_fname}", f"atlases/cobralab/{_fname}", _sha), **UKB_TEMPLATE},
        mni_aligned=False,
        citation=ATLASES["cobralab_patho"].citation, url=ATLASES["cobralab_patho"].url,
        notes="Released only in the CoBrALab UK Biobank template space (no transform shipped). Run "
              "`wmhparc align-template cobralab` (ANTs SyN of the UKB template T1 to MNI) to obtain an MNI version.",
    )

ATLASES["bullseye36"] = Atlas(
    name="bullseye36",
    description="36-region bullseye (4 lobes x 2 hemispheres + BGIT) x 4 depth layers, built in MNI152NLin6Asym "
                "from the Harvard-Oxford atlases",
    space="MNI152NLin6Asym", lut="bullseye36", builder="build_mni_bullseye36",
    citation="Sudre C.H. et al. Bullseye's representation of cerebral white matter hyperintensities. "
             "J Neuroradiol 2018; Sanroma G. bullseye_pipeline (github.com/gsanroma/bullseye_pipeline).",
    notes="Depth layers = quartiles of normalised distance ventricles->cortex; lobes propagated into WM from the "
          "nearest cortical lobe label (insula counted as frontal, cingulate filled from neighbours), basal "
          "ganglia + thalamus override the lobes (BGIT). Label order follows the Edinburgh auto-wmh-bullseye-parc tool.",
)
ATLASES["bullseye20"] = Atlas(
    name="bullseye20",
    description="bullseye36 folded over hemispheres and numbered like the Chung 20-ROI atlas",
    space="MNI152NLin6Asym", lut="bullseye20", builder="build_mni_bullseye20",
    citation=ATLASES["bullseye36"].citation,
    notes="Same label numbering as chung20 (1-4 BGIT, 5-8 frontal, 9-12 occipital, 13-16 temporal, 17-20 parietal).",
)


def get_atlas(name: str) -> Atlas:
    try:
        return ATLASES[name]
    except KeyError:
        raise KeyError(f"Unknown atlas {name!r}; choose from {sorted(ATLASES)}") from None


def load_lut(name: str) -> pd.DataFrame:
    """Load a label table (TSV with at least ``index`` and ``name`` columns)."""
    if name in ATLASES:
        name = ATLASES[name].lut
    with ir.as_file(ir.files("wmhparc").joinpath("data", "luts", f"{name}.tsv")) as p:
        if not p.exists():
            raise FileNotFoundError(f"No LUT named {name!r}")
        return pd.read_csv(p, sep="\t")


def fetch_atlas(name: str, force: bool = False, quiet: bool = False) -> Dict[str, Path]:
    """Download all resources of a released atlas; returns {key: path}."""
    atlas = get_atlas(name)
    return {k: r.fetch(force=force, quiet=quiet) for k, r in atlas.resources.items()}


def derived_dir() -> Path:
    d = cache_dir() / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def atlas_path(name: str, space: Optional[str] = None, build: bool = True, quiet: bool = False) -> Path:
    """Return the path of the label image for ``name``.

    Released MNI-aligned atlases are downloaded.  Built atlases are built (and cached) on
    first use.  UKB-space CoBrALab atlases need ``align_template`` to have been run; then
    the MNI version is returned.
    """
    atlas = get_atlas(name)
    if atlas.is_built:
        out = derived_dir() / f"{name}_space-{atlas.space}_dseg.nii.gz"
        if not out.exists():
            if not build:
                raise FileNotFoundError(f"{out} not built yet; run `wmhparc build-bullseye-mni`")
            from . import bullseye  # local import to keep import time low
            getattr(bullseye, atlas.builder)(out, quiet=quiet)
        return out
    if atlas.mni_aligned:
        return atlas.resources[atlas.label_key].fetch(quiet=quiet)
    # foreign-template atlas: look for an aligned copy
    target = space or "MNI152NLin2009cAsym"
    aligned = derived_dir() / f"{name}_space-{target}_dseg.nii.gz"
    if not aligned.exists():
        raise FileNotFoundError(
            f"{name} is released in {atlas.space}; no MNI-aligned copy found at {aligned}. "
            f"Run `wmhparc align-template cobralab --template {target}` first.")
    return aligned


def load_atlas(name: str, space: Optional[str] = None, quiet: bool = False) -> Tuple[nib.Nifti1Image, pd.DataFrame]:
    """Return (label image, LUT) for an atlas, fetching or building it if required."""
    from .imageio import load_labels
    p = atlas_path(name, space=space, quiet=quiet)
    data, affine, header = load_labels(p)
    img = nib.Nifti1Image(data.astype(np.int16), affine)
    return img, load_lut(name)


def describe(name: str) -> str:
    a = get_atlas(name)
    lut = load_lut(name)
    lines = [f"{a.name}: {a.description}", f"  space:    {a.space}", f"  citation: {a.citation}"]
    if a.url:
        lines.append(f"  url:      {a.url}")
    if a.notes:
        lines.append(f"  notes:    {a.notes}")
    lines.append("  labels:")
    lines.append("    " + lut.to_string(index=False).replace("\n", "\n    "))
    return "\n".join(lines)


def list_atlases() -> List[str]:
    return sorted(ATLASES)
