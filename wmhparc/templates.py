"""Standard-space templates (fetched from TemplateFlow's public S3 bucket).

Two MNI flavours are supported because the released parcellations live in
different ones:

* ``MNI152NLin2009cAsym`` (ICBM 2009c): the CoBrALab ``WMH_parc_patho_MNI``
  parcellation is released on exactly this 193x229x193 1 mm grid.
* ``MNI152NLin6Asym`` (FSL's MNI152): the Harvard-Oxford atlases used to build
  the 36-region bullseye are released on this 182x218x182 1 mm grid.

The Chung 20-ROI atlas is on a 0.5 mm Colin27 (``ch2better``) grid that is
affinely aligned to MNI152; see :mod:`wmhparc.atlases`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from .cache import fetch_file

TF_URL = "https://templateflow.s3.amazonaws.com"


@dataclass(frozen=True)
class Resource:
    url: str
    sha256: str
    relpath: str

    def fetch(self, force: bool = False, quiet: bool = False) -> Path:
        return fetch_file(self.url, self.relpath, self.sha256, force=force, quiet=quiet)


def _tf(tpl: str, fname: str, sha: str) -> Resource:
    return Resource(f"{TF_URL}/tpl-{tpl}/{fname}", sha, f"templates/{tpl}/{fname}")


@dataclass(frozen=True)
class Template:
    name: str
    files: Dict[str, Resource] = field(default_factory=dict)
    description: str = ""

    def fetch(self, key: str, **kw) -> Path:
        return self.files[key].fetch(**kw)


TEMPLATES: Dict[str, Template] = {
    "MNI152NLin2009cAsym": Template(
        name="MNI152NLin2009cAsym",
        description="ICBM 152 nonlinear asymmetric 2009c, 1 mm, 193x229x193 (Fonov et al. 2011)",
        files={
            "T1w": _tf("MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
                       "1f27aabea9f7183dc0c69dafa71e1787b921ca778d655d9c1bf302b273d5627a"),
            "brain": _tf("MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz",
                         "97cee2cf7f51388d76d7b3632c8375d688d0fa171e1bede81fcef23726bb3d73"),
            "mask": _tf("MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz",
                        "e40bb1816736504d4c25abc243c1c8503df1d308f64c048318aa4b829e09a7ae"),
            "WM": _tf("MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_res-01_label-WM_probseg.nii.gz",
                      "b2f80e29f5a1ef55325215d1716ef611f5b1a3cd97b4a49ef4e9564e9564e945"),
            "GM": _tf("MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz",
                      "662b18e83dddc554b19c621d9750af3454b54d4e103df03633eacced3884805a"),
            "CSF": _tf("MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_res-01_label-CSF_probseg.nii.gz",
                       "8a74e36b067a3fb670f6785b94479cc3df4d219a3f72d69d57aef332e4947048"),
        },
    ),
    "MNI152NLin6Asym": Template(
        name="MNI152NLin6Asym",
        description="MNI152 nonlinear 6th generation (FSL MNI152), 1 mm, 182x218x182",
        files={
            "T1w": _tf("MNI152NLin6Asym", "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz",
                       "f675946c0a077e0682885c3d6085f10acb0dd22b2274a6cbbf04969c08a77258"),
            "brain": _tf("MNI152NLin6Asym", "tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz",
                         "72f8edc99d7e696da7717db20b73897d8da5909d7f30b1c9a7e2a79e9f64ff2f"),
            "mask": _tf("MNI152NLin6Asym", "tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz",
                        "6d540075dac093fa5070a929dc5ea52795d54cc9ab57fe1272086dc70314b428"),
            # Harvard-Oxford maximum-probability atlases (threshold 0), lateralised cortical + subcortical
            "HOCPAL": _tf("MNI152NLin6Asym", "tpl-MNI152NLin6Asym_res-01_atlas-HOCPAL_desc-th0_dseg.nii.gz",
                          "92897760d6a0cbcc85bcfc6294b437ee5e551a2ae83b7b47fe88ba7440ba5534"),
            "HOSPA": _tf("MNI152NLin6Asym", "tpl-MNI152NLin6Asym_res-01_atlas-HOSPA_desc-th0_dseg.nii.gz",
                         "91fc2b6231fadec5eefd3f1a827688722395dc2d878a79f5dc64d2d1362fe1fe"),
        },
    ),
}

DEFAULT_TEMPLATE = "MNI152NLin2009cAsym"


def get_template(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise KeyError(f"Unknown template {name!r}; choose from {sorted(TEMPLATES)}") from None
