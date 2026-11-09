# wmhparc

Regional masks of white-matter hyperintensities (WMH) from released MNI-space parcellations,
for studying WMH distribution and its pathophysiology.

Given a subject's WMH segmentation (and a T1 and/or FLAIR), `wmhparc`:

1. registers the subject to MNI with ANTs SyN (T1 → MNI, with a rigid FLAIR → T1 step composed in;
   FLAIR-only SyN with mutual information as the fallback),
2. brings each parcellation into the subject's FLAIR grid (or the WMH mask into MNI),
3. intersects it with the WMH mask and writes per-region masks, a region-labelled WMH map, and a
   TSV of regional volumes.

Everything except the registration is pure numpy/scipy/nibabel. Registration uses ANTsPy
(`pip install antspyx`) or the ANTs binaries if they are on `$PATH`.

## Parcellations

| name | what | space | labels |
|---|---|---|---|
| `chung20` | Chung et al. 2025 bullseye-style atlas, 5 regions × 4 depth layers | Colin27 (ch2better) 0.5 mm grid, MNI152-aligned | 20 |
| `cobralab_patho` | CoBrALab data-driven WMH "pathophysiology" clusters (Parent et al. 2026) | MNI152NLin2009cAsym 1 mm | 3 |
| `cobralab_lobar`, `cobralab_pvdeep`, `cobralab_vascular`, `cobralab_fiber` | CoBrALab comparison parcellations | CoBrALab UK Biobank template (needs one alignment step, see below) | 6 / 2 / 5 / 3 |
| `bullseye36` | 36-region bullseye (Sudre et al. 2018: 4 lobes × 2 hemispheres + BGIT, × 4 layers), **built here** from the Harvard-Oxford atlases | MNI152NLin6Asym 1 mm | 36 |
| `bullseye20` | `bullseye36` folded over hemispheres, numbered like `chung20` | MNI152NLin6Asym 1 mm | 20 |

`wmhparc info <name>` prints provenance notes and the label table; the tables live in
`wmhparc/data/luts/*.tsv`.

### Chung 20-ROI atlas: label table read off the file

The Chung repository ships no lookup table, so the meaning of each ROI was inferred from the
image itself (and is checked by `tests/test_atlases_real.py`):

* labels form five chains of four mutually adjacent ROIs (1-4, 5-8, 9-12, 13-16, 17-20); the
  chains are concentric (label k touches only k±1 within a chain);
* the first label of each chain is the smallest, hugs the lateral ventricles and touches the first
  labels of the other chains → **layer 1 = periventricular**; the fourth is the largest and reaches
  the cortex → **layer 4 = juxtacortical** (quartiles of the normalised ventricle→cortex distance);
* region identity from bounding boxes in MNI mm: 1-4 central (**BGIT**, basal ganglia + thalamus),
  5-8 reach y > 60 (**frontal**), 9-12 reach y < -95 (**occipital**), 13-16 reach z < -40
  (**temporal**), 17-20 reach z > 75 (**parietal**). There is no hemisphere split.

So `label = 4 × region + layer` with regions ordered BGIT, frontal, occipital, temporal, parietal.
As a cross-check, parcellating the same lesion with `chung20` and the independently built
`bullseye20` gives near-identical regional distributions.

### CoBrALab labels

From `analyses/2_spatial_clust/2e_overlap_other_parc.R` in the release: `WMH_parc_patho_MNI`
1 = periventricular, 2 = posterior, 3 = anterior; lobar 1 frontal / 3 temporal / 5 parietal /
7 cerebellum / 9 occipital / 11 brainstem; PV/deep 2 periventricular / 1 deep; vascular 1 ACA / 3
MCA / 5 PCA / 7 VB / 9 ventricles; fibre 1 association / 2 commissural / 3 projection. Note that
the pathophysiology clusters only cover voxels where WMH are prevalent in UK Biobank; WMH outside
them are reported as `unlabelled` (use `--fill-mm` to assign them to the nearest cluster, which is
what the authors' `2d_assign_excluded_voxels.R` does).

### 36-region bullseye in MNI (built locally)

Nobody has released a 36-region bullseye in MNI, so `wmhparc build-bullseye-mni` builds one from
TemplateFlow's Harvard-Oxford subcortical (`HOSPA`) and lateralised cortical (`HOCPAL`) maximum
probability atlases in MNI152NLin6Asym, following Sanroma's `bullseye_pipeline` (the one Chung et
al. used):

* depth `d = D_vent / (D_vent + D_cortex)` (Euclidean distances to the lateral ventricles and to
  the lobar cortex, insula and cingulate excluded as in the original pipeline), four equal-width
  layers;
* lobes propagated into the white matter of the same hemisphere from the nearest cortical label
  (Harvard-Oxford structure → lobe mapping in `wmhparc/data/harvard_oxford_cortical_lobes.tsv`;
  insula counted as frontal, cingulate WM filled from its neighbours);
* basal ganglia + thalamus override the lobes as the BGIT region.

The same code builds a **subject-space** bullseye from a FreeSurfer/FastSurfer `aparc+aseg`
(`wmhparc build-bullseye --aparc-aseg ...`) or from SynthSeg output plus a lobe image
(`wmhparc.bullseye.inputs_from_synthseg`), if you prefer the Sanroma/Edinburgh route over
registration.

## Install

```bash
pip install -e .            # core (numpy, scipy, nibabel, pandas)
pip install antspyx         # registration backend (or have ANTs binaries on $PATH)
wmhparc fetch all           # ~70 MB of atlases + templates into $WMHPARC_HOME (default ~/.cache/wmhparc)
```

Downloads are SHA-256 pinned; a changed upstream file raises instead of being used silently.

## Usage

One shot (register + parcellate with the default atlases):

```bash
wmhparc run --t1 sub-01_T1w.nii.gz --flair sub-01_FLAIR.nii.gz \
    --wmh sub-01_FLAIR_wmhseg.nii.gz --out derivatives/wmhparc/sub-01 --prefix sub-01 \
    --atlas chung20 cobralab_patho bullseye36 bullseye20 --split-masks --threads 8
```

Step by step:

```bash
wmhparc register --t1 sub-01_T1w.nii.gz --flair sub-01_FLAIR.nii.gz --out sub-01/xfm
```

```bash
wmhparc parcellate --wmh sub-01_FLAIR_wmhseg.nii.gz --xfm sub-01/xfm --atlas chung20 bullseye36 \
    --out sub-01 --prefix sub-01 --split-masks
```

FLAIR only: `wmhparc register --flair ...` (SyN with mutual information). WMH mask already in MNI:
omit `--xfm` (header-based resampling only). Group analysis in template space: `--space template`.

CoBrALab's UKB-space parcellations need one registration of their template to MNI (done once,
cached):

```bash
wmhparc align-template cobralab --template MNI152NLin2009cAsym
```

Build a subject-space bullseye from FreeSurfer instead of registration:

```bash
mri_convert sub-01/mri/aparc+aseg.mgz aparc+aseg.nii.gz
wmhparc build-bullseye --aparc-aseg aparc+aseg.nii.gz --out sub-01/sub-01
```

### Outputs (per subject, per atlas)

```
<prefix>_atlas-<name>_space-native_dseg.nii.gz                 atlas resampled onto the WMH grid
<prefix>_atlas-<name>_space-native_desc-wmhparc_dseg.nii.gz    WMH voxels carrying their region label
<prefix>_atlas-<name>_space-native_volumes.tsv                 per-region table (see below)
<prefix>_atlas-<name>_space-native_masks/..._label-NN_desc-<region>_mask.nii.gz   (with --split-masks)
<prefix>_desc-allatlases_volumes.tsv                           all atlases, long format
```

The volume table has, per region: `wmh_voxels`, `wmh_volume_ml`, `region_volume_ml` (atlas region
in subject space), `wmh_fraction_of_region` (regional WMH load) and `fraction_of_total_wmh`
(spatial distribution), plus an `unlabelled` row for WMH outside the atlas.

Python: see `examples/python_api.py` (and `examples/monai_regional_wmh.py` for using the maps in a
MONAI pipeline). The core objects are `wmhparc.registration.register_subject` →
`TransformChain`, `wmhparc.parcellate.parcellate_wmh`, `wmhparc.bullseye.build_bullseye`.

## Registration caveat (read this)

The periventricular layers track ventricle size. A subject with enlarged ventricles gets the wrong
depth labels if the warp is poor, and large confluent WMH can pull a FLAIR-only deformable warp.
Therefore:

* register with **ANTs SyN**, preferably **T1 → MNI**, with a **rigid FLAIR → T1** step composed
  in (this is what `register --t1 --flair` does: CC metric for T1 → template as in
  `antsRegistrationSyN.sh -t s`, mutual information for the rigid step);
* use the T1 whenever you have it;
* if only FLAIR is available, SyN on FLAIR with mutual information is still acceptable because the
  CSF contrast is strong (`register --flair`);
* check the warped ventricles: `<out>/xfm/*Warped.nii.gz` (ANTs CLI) or warp a ventricle mask
  with `wmhparc.registration.apply_transforms` and overlay it on the template.
* `--quick` uses the `antsRegistrationSyNQuick` settings; use the default (full) settings for
  final numbers.

Template spaces: registration defaults to `MNI152NLin2009cAsym` (the space of the CoBrALab
parcellation). The Chung atlas is on a Colin27 grid that is affinely aligned to MNI152 and the
bullseye is built in `MNI152NLin6Asym`; both are treated as MNI152 at the lobe/quartile-layer
granularity of these parcellations. ANTs resamples through physical coordinates, so the differing
grids (0.5 mm / 1 mm, different field of view) are handled transparently. If you want to remove even
the residual template-to-template difference, register the atlas' own template to your target with
`wmhparc align-template <template_T1.nii.gz> --labels <atlas.nii.gz>`.

## Transform conventions

`TransformChain` stores the ANTs products and gives the lists in ANTs order (fixed side first):

* atlas (MNI) → FLAIR grid: `[FLAIR→T1 rigid (inverted), T1→MNI affine (inverted), T1→MNI inverse warp]`
* WMH (FLAIR) → MNI grid: `[T1→MNI warp, T1→MNI affine, FLAIR→T1 rigid]`

Labels are resampled with ANTs `GenericLabel` (native) / nearest neighbour.

## Tests

```bash
pytest
```

Pure-synthetic tests always run; tests on the real atlases run when they are in the cache
(`wmhparc fetch all`). The registration path was validated with ANTsPy on a synthetic subject
(MNI152NLin6Asym head as "T1", a rigidly displaced contrast-inverted copy as "FLAIR"): the rigid
step recovers the displacement (brain-mask Dice 0.998) and T1 → MNI152NLin2009cAsym SyN gives a
brain-mask Dice of 0.97 with `--quick` at 2 mm.

## References

* Chung J. et al. Distinct spatiotemporal patterns of white matter hyperintensity progression. *Nat Commun* 2025. Atlas: https://github.com/JinyongChung/WMH_progression_modeling
* Parent O. et al. Characterizing spatiotemporal white matter hyperintensity pathophysiology in vivo to disentangle vascular and neurodegenerative contributions. *Nat Commun* 2026. Data: https://github.com/CoBrALab/WMH_patho_UKB
* Sudre C.H. et al. Bullseye's representation of cerebral white matter hyperintensities. *J Neuroradiol* 2018. Pipeline: https://github.com/gsanroma/bullseye_pipeline
* Edinburgh automated bullseye tool (label ordering of `bullseye36`): https://github.com/BenjaminPhi5/auto-wmh-bullseye-parc
* Harvard-Oxford atlases and MNI templates via TemplateFlow (https://www.templateflow.org); Fonov V. et al. *NeuroImage* 2011.
* Avants B.B. et al. ANTs SyN. *Med Image Anal* 2008.
