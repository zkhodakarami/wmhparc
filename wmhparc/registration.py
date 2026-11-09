"""Subject-to-template registration with ANTs (SyN) and transform bookkeeping.

Two interchangeable backends:

* ``antspy`` -- the ANTsPy python package (``pip install antspyx``).
* ``cli``    -- the ANTs command-line tools (``antsRegistration``, ``antsApplyTransforms``) on ``$PATH``.

Recommended workflow (see README, "Registration caveat"):

* T1 available: **T1 -> template with SyN** (CC metric, as ``antsRegistrationSyN.sh -t s``),
  plus a **rigid FLAIR -> T1** step composed in, so that WMH masks drawn on FLAIR can be
  mapped through T1.  Large confluent WMH can pull a FLAIR-only deformable warp; the T1 is
  the safer anatomy to drive the non-linear step and the periventricular layers of the
  bullseye are only as good as the ventricle alignment.
* FLAIR only: **FLAIR -> template with SyN** using mutual information (cross-modal), which
  is acceptable because CSF/ventricle contrast is strong on FLAIR.

Transform ordering follows the ANTs convention: the list passed to ``antsApplyTransforms``
reads from the *fixed* (reference) side to the *moving* side, so to resample an MNI atlas
onto a FLAIR grid the list is ``[flair->t1 rigid (inverted), t1->MNI affine (inverted),
t1->MNI inverse warp]`` and to warp a FLAIR-space WMH mask into MNI it is
``[t1->MNI warp, t1->MNI affine, flair->t1 rigid]``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .templates import DEFAULT_TEMPLATE, get_template

PathLike = Union[str, Path]
Interp = str  # "nearest" | "linear" | "label"


# ----------------------------------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------------------------------
def have_antspy() -> bool:
    try:
        import importlib.util
        if importlib.util.find_spec("ants") is None:
            return False
        import ants  # noqa: F401
        return True
    except Exception:
        return False


def have_ants_cli() -> bool:
    return shutil.which("antsRegistration") is not None and shutil.which("antsApplyTransforms") is not None


def resolve_backend(backend: str = "auto") -> str:
    if backend == "auto":
        if have_antspy():
            return "antspy"
        if have_ants_cli():
            return "cli"
        raise RuntimeError("No ANTs backend found: `pip install antspyx` or put antsRegistration on $PATH")
    if backend == "antspy" and not have_antspy():
        raise RuntimeError("backend 'antspy' requested but `import ants` failed (pip install antspyx)")
    if backend == "cli" and not have_ants_cli():
        raise RuntimeError("backend 'cli' requested but antsRegistration/antsApplyTransforms are not on $PATH")
    if backend not in ("antspy", "cli"):
        raise ValueError(f"unknown backend {backend!r}")
    return backend


# ----------------------------------------------------------------------------------------------
# Transform chain
# ----------------------------------------------------------------------------------------------
@dataclass
class TransformChain:
    """Registration products of one subject, serialisable to JSON.

    ``anat`` is the image that drove the non-linear registration (T1 if given, else FLAIR).
    """
    template: str
    template_image: str
    anat: str
    anat_modality: str                           # "T1w" or "FLAIR"
    anat_to_template_warp: str                   # ...1Warp.nii.gz
    anat_to_template_inverse_warp: str           # ...1InverseWarp.nii.gz
    anat_to_template_affine: str                 # ...0GenericAffine.mat
    flair: Optional[str] = None
    flair_to_t1_rigid: Optional[str] = None      # ...0GenericAffine.mat (fixed = T1, moving = FLAIR)
    backend: str = ""
    extra: dict = field(default_factory=dict)

    # --- ordering helpers --------------------------------------------------------------------
    def to_template(self, from_space: str = "anat") -> Tuple[List[str], List[bool]]:
        """Transform list to resample an image living in ``from_space`` ('anat'|'flair') onto the
        template grid (fixed = template, moving = subject image)."""
        xf = [self.anat_to_template_warp, self.anat_to_template_affine]
        inv = [False, False]
        if from_space == "flair" and self.flair_to_t1_rigid:
            xf.append(self.flair_to_t1_rigid)
            inv.append(False)
        elif from_space not in ("anat", "flair"):
            raise ValueError(from_space)
        return xf, inv

    def from_template(self, to_space: str = "anat") -> Tuple[List[str], List[bool]]:
        """Transform list to resample a template-space image (atlas) onto the subject grid
        (fixed = subject image, moving = atlas)."""
        xf: List[str] = []
        inv: List[bool] = []
        if to_space == "flair" and self.flair_to_t1_rigid:
            xf.append(self.flair_to_t1_rigid)
            inv.append(True)
        elif to_space not in ("anat", "flair"):
            raise ValueError(to_space)
        xf += [self.anat_to_template_affine, self.anat_to_template_inverse_warp]
        inv += [True, False]
        return xf, inv

    def space_of(self, modality: str) -> str:
        """'flair' if a separate FLAIR->T1 rigid exists and modality is FLAIR, else 'anat'."""
        if modality.lower() == "flair" and self.flair_to_t1_rigid:
            return "flair"
        return "anat"

    # --- persistence ---------------------------------------------------------------------------
    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TransformChain":
        path = Path(path)
        if path.is_dir():
            path = path / "transforms.json"
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


# ----------------------------------------------------------------------------------------------
# ANTs command-line recipes (mirroring antsRegistrationSyN.sh)
# ----------------------------------------------------------------------------------------------
def ants_registration_command(fixed: PathLike, moving: PathLike, prefix: PathLike, kind: str = "syn",
                              syn_metric: str = "CC", quick: bool = False,
                              fixed_mask: Optional[PathLike] = None, moving_mask: Optional[PathLike] = None,
                              ) -> List[str]:
    """Build an ``antsRegistration`` command.

    ``kind``: 'rigid' | 'affine' | 'syn'.  ``syn_metric``: 'CC' (same modality) or 'MI' (cross-modal).
    Parameters follow ``antsRegistrationSyN.sh`` (``-t s``/``-t r``; ``quick`` -> ``antsRegistrationSyNQuick.sh``).
    """
    fixed, moving, prefix = str(fixed), str(moving), str(prefix)
    cmd = ["antsRegistration", "--verbose", "0", "--dimensionality", "3", "--float", "0",
           "--collapse-output-transforms", "1",
           "--output", f"[{prefix},{prefix}Warped.nii.gz,{prefix}InverseWarped.nii.gz]",
           "--interpolation", "Linear", "--use-histogram-matching", "0",
           "--winsorize-image-intensities", "[0.005,0.995]",
           "--initial-moving-transform", f"[{fixed},{moving},1]"]
    lin_conv = "[1000x500x250x0,1e-6,10]" if quick else "[1000x500x250x100,1e-6,10]"
    cmd += ["--transform", "Rigid[0.1]", "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
            "--convergence", lin_conv, "--shrink-factors", "8x4x2x1", "--smoothing-sigmas", "3x2x1x0vox"]
    if kind in ("affine", "syn"):
        cmd += ["--transform", "Affine[0.1]", "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
                "--convergence", lin_conv, "--shrink-factors", "8x4x2x1", "--smoothing-sigmas", "3x2x1x0vox"]
    if kind == "syn":
        if syn_metric.upper() == "CC":
            metric = f"CC[{fixed},{moving},1,4]"
        else:
            metric = f"MI[{fixed},{moving},1,32,Regular,0.25]"
        syn_conv = "[100x70x50x0,1e-6,10]" if quick else "[100x70x50x20,1e-6,10]"
        cmd += ["--transform", "SyN[0.1,3,0]", "--metric", metric,
                "--convergence", syn_conv, "--shrink-factors", "8x4x2x1", "--smoothing-sigmas", "3x2x1x0vox"]
    if fixed_mask or moving_mask:
        cmd += ["--masks", f"[{fixed_mask or 'NULL'},{moving_mask or 'NULL'}]"]
    if kind not in ("rigid", "affine", "syn"):
        raise ValueError(kind)
    return cmd


def ants_apply_command(fixed: PathLike, moving: PathLike, out: PathLike, transforms: Sequence[str],
                       invert: Sequence[bool], interpolation: Interp = "nearest") -> List[str]:
    interp = {"nearest": "NearestNeighbor", "linear": "Linear", "label": "GenericLabel[Linear]",
              "bspline": "BSpline[3]"}[interpolation]
    cmd = ["antsApplyTransforms", "-d", "3", "-i", str(moving), "-r", str(fixed), "-o", str(out), "-n", interp]
    for t, inv in zip(transforms, invert):
        cmd += ["-t", f"[{t},1]" if inv else str(t)]
    return cmd


def _run(cmd: Sequence[str], threads: Optional[int] = None, quiet: bool = False) -> None:
    env = os.environ.copy()
    if threads:
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    if not quiet:
        print("[wmhparc] $ " + " ".join(cmd))
    subprocess.run(list(cmd), check=True, env=env)


# ----------------------------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------------------------
def _cli_outputs(prefix: Path, kind: str) -> Tuple[Path, Optional[Path], Optional[Path]]:
    aff = Path(f"{prefix}0GenericAffine.mat")
    if kind == "syn":
        return aff, Path(f"{prefix}1Warp.nii.gz"), Path(f"{prefix}1InverseWarp.nii.gz")
    return aff, None, None


def register(fixed: PathLike, moving: PathLike, prefix: PathLike, kind: str = "syn", syn_metric: str = "CC",
             backend: str = "auto", quick: bool = False, threads: Optional[int] = None,
             fixed_mask: Optional[PathLike] = None, quiet: bool = False
             ) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Run a registration and return ``(affine.mat, warp, inverse_warp)`` (warps are None for rigid/affine)."""
    backend = resolve_backend(backend)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if backend == "cli":
        cmd = ants_registration_command(fixed, moving, prefix, kind=kind, syn_metric=syn_metric, quick=quick,
                                        fixed_mask=fixed_mask)
        _run(cmd, threads=threads, quiet=quiet)
        aff, warp, iwarp = _cli_outputs(prefix, kind)
        for p in (aff, warp, iwarp):
            if p is not None and not p.exists():
                raise RuntimeError(f"antsRegistration did not produce {p}")
        return aff, warp, iwarp
    # antspy
    import ants
    if threads:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    fx = ants.image_read(str(fixed))
    mv = ants.image_read(str(moving))
    kw = {}
    if fixed_mask is not None:
        kw["mask"] = ants.image_read(str(fixed_mask))
    if kind == "rigid":
        tot = "antsRegistrationSyNQuick[r]" if quick else "antsRegistrationSyN[r]"
    elif kind == "affine":
        tot = "antsRegistrationSyNQuick[a]" if quick else "antsRegistrationSyN[a]"
    elif kind == "syn":
        if syn_metric.upper() == "CC":
            tot = "antsRegistrationSyNQuick[s]" if quick else "antsRegistrationSyN[s]"
        else:
            # Mattes MI at every stage: the right choice for FLAIR -> T1-weighted template
            tot = "SyN"
            kw.update(aff_metric="mattes", syn_metric="mattes",
                      reg_iterations=(40, 20, 0) if quick else (100, 70, 50, 20))
    else:
        raise ValueError(kind)
    if not quiet:
        print(f"[wmhparc] ants.registration(type_of_transform={tot!r}) fixed={fixed} moving={moving}")
    reg = ants.registration(fixed=fx, moving=mv, type_of_transform=tot, outprefix=str(prefix), verbose=False, **kw)
    fwd = [Path(p) for p in reg["fwdtransforms"]]
    inv = [Path(p) for p in reg["invtransforms"]]
    aff = next(p for p in fwd if p.suffix == ".mat")
    warp = next((p for p in fwd if "Warp" in p.name and "Inverse" not in p.name), None)
    iwarp = next((p for p in inv if "InverseWarp" in p.name), None)
    # antspy may write transforms to temp names; keep stable copies next to prefix
    aff = _stabilise(aff, Path(f"{prefix}0GenericAffine.mat"))
    if warp is not None:
        warp = _stabilise(warp, Path(f"{prefix}1Warp.nii.gz"))
        iwarp = _stabilise(iwarp, Path(f"{prefix}1InverseWarp.nii.gz"))
    return aff, warp, iwarp


def _stabilise(src: Path, dst: Path) -> Path:
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst


def apply_transforms(fixed: PathLike, moving: PathLike, out: PathLike, transforms: Sequence[str],
                     invert: Sequence[bool], interpolation: Interp = "nearest", backend: str = "auto",
                     quiet: bool = True) -> Path:
    """Resample ``moving`` onto the grid of ``fixed`` through ``transforms`` (ANTs ordering)."""
    backend = resolve_backend(backend)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if backend == "cli":
        _run(ants_apply_command(fixed, moving, out, transforms, invert, interpolation), quiet=quiet)
        return out
    import ants
    interp = {"nearest": "nearestNeighbor", "linear": "linear", "label": "genericLabel", "bspline": "bSpline"}[interpolation]
    fx = ants.image_read(str(fixed))
    mv = ants.image_read(str(moving))
    res = ants.apply_transforms(fixed=fx, moving=mv, transformlist=[str(t) for t in transforms],
                                whichtoinvert=list(invert), interpolator=interp)
    ants.image_write(res, str(out))
    return out


def register_subject(t1: Optional[PathLike] = None, flair: Optional[PathLike] = None,
                     template: str = DEFAULT_TEMPLATE, out_dir: PathLike = "wmhparc_reg",
                     backend: str = "auto", skull_stripped: bool = False, quick: bool = False,
                     threads: Optional[int] = None, use_template_mask: bool = False,
                     quiet: bool = False) -> TransformChain:
    """Register a subject to a standard template and write ``transforms.json``.

    * ``t1`` and ``flair``: SyN T1->template (CC) + rigid FLAIR->T1 (MI).
    * ``t1`` only: SyN T1->template.
    * ``flair`` only: SyN FLAIR->template with mutual information.

    ``skull_stripped``: the inputs are brain-extracted, so register to the template *brain*.
    ``use_template_mask``: restrict the metric to the template brain mask (helps with whole-head
    inputs that have very different neck/face coverage).
    """
    if t1 is None and flair is None:
        raise ValueError("give at least one of t1 / flair")
    tpl = get_template(template)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl_img = tpl.fetch("brain" if skull_stripped else "T1w", quiet=quiet)
    tpl_mask = tpl.fetch("mask", quiet=quiet) if use_template_mask else None
    backend = resolve_backend(backend)

    if t1 is not None:
        anat, modality, metric = Path(t1), "T1w", "CC"
    else:
        anat, modality, metric = Path(flair), "FLAIR", "MI"
    prefix = out_dir / f"{modality}_to_{template}_"
    aff, warp, iwarp = register(tpl_img, anat, prefix, kind="syn", syn_metric=metric, backend=backend,
                                quick=quick, threads=threads, fixed_mask=tpl_mask, quiet=quiet)
    chain = TransformChain(template=template, template_image=str(tpl_img), anat=str(anat), anat_modality=modality,
                           anat_to_template_warp=str(warp), anat_to_template_inverse_warp=str(iwarp),
                           anat_to_template_affine=str(aff), backend=backend,
                           extra={"quick": quick, "skull_stripped": skull_stripped})
    if t1 is not None and flair is not None:
        rprefix = out_dir / "FLAIR_to_T1w_"
        raff, _, _ = register(t1, flair, rprefix, kind="rigid", backend=backend, quick=quick, threads=threads,
                              quiet=quiet)
        chain.flair = str(Path(flair))
        chain.flair_to_t1_rigid = str(raff)
    chain.save(out_dir / "transforms.json")
    if not quiet:
        print(f"[wmhparc] wrote {out_dir / 'transforms.json'}")
    return chain


# ----------------------------------------------------------------------------------------------
# Template-to-template alignment (e.g. CoBrALab UK Biobank template -> MNI)
# ----------------------------------------------------------------------------------------------
def align_template(template_t1: PathLike, target: str, out_dir: PathLike, label_images: Sequence[PathLike] = (),
                   template_mask: Optional[PathLike] = None, backend: str = "auto", quick: bool = False,
                   threads: Optional[int] = None, quiet: bool = False) -> Tuple[TransformChain, List[Path]]:
    """SyN-register a foreign template's T1 to a TemplateFlow template and warp label images along.

    Returns the chain and the list of warped label images (nearest-neighbour), named
    ``<stem>_space-<target>_dseg.nii.gz`` inside ``out_dir``.
    """
    out_dir = Path(out_dir)
    tpl = get_template(target)
    skull_stripped = template_mask is not None
    moving = Path(template_t1)
    if template_mask is not None:
        # multiply mask in so we register brain to brain
        import nibabel as nib
        import numpy as np
        img = nib.load(str(moving))
        m = np.asarray(nib.load(str(template_mask)).dataobj) > 0
        moving = out_dir / (moving.name.replace(".nii.gz", "").replace(".nii", "") + "_brain.nii.gz")
        out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(img.dataobj) * m, img.affine), str(moving))
    chain = register_subject(t1=moving, template=target, out_dir=out_dir, backend=backend,
                             skull_stripped=skull_stripped, quick=quick, threads=threads, quiet=quiet)
    fixed = tpl.fetch("T1w", quiet=True)
    warped: List[Path] = []
    xf, inv = chain.to_template("anat")
    for lab in label_images:
        lab = Path(lab)
        stem = lab.name.replace(".nii.gz", "").replace(".nii", "")
        out = out_dir / f"{stem}_space-{target}_dseg.nii.gz"
        apply_transforms(fixed, lab, out, xf, inv, interpolation="label", backend=backend, quiet=quiet)
        warped.append(out)
    return chain, warped
