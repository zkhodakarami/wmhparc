"""Minimal Python-API walkthrough (same steps as `wmhparc run`).

    python examples/python_api.py --t1 sub-01_T1w.nii.gz --flair sub-01_FLAIR.nii.gz \
        --wmh sub-01_FLAIR_wmhseg.nii.gz --out derivatives/wmhparc/sub-01
"""
import argparse

from wmhparc.parcellate import parcellate_many
from wmhparc.registration import register_subject

ap = argparse.ArgumentParser()
ap.add_argument("--t1")
ap.add_argument("--flair")
ap.add_argument("--wmh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

# 1. ANTs SyN T1 -> MNI152NLin2009cAsym (+ rigid FLAIR -> T1 composed in); FLAIR-only if no T1
chain = register_subject(t1=args.t1, flair=args.flair, template="MNI152NLin2009cAsym",
                         out_dir=f"{args.out}/xfm", quick=args.quick)

# 2. bring each atlas into the FLAIR grid, intersect with the WMH mask, write masks + volume tables
results = parcellate_many(args.wmh, ["chung20", "cobralab_patho", "bullseye36", "bullseye20"], chain,
                          out_dir=args.out, prefix="sub-01", space="native", wmh_modality="FLAIR",
                          split_masks=True)
for r in results:
    df = r.volumes
    print(r.atlas_name, "total WMH %.2f ml" % df.attrs["total_wmh_volume_ml"])
    print(df[["index", "name", "wmh_volume_ml", "fraction_of_total_wmh"]].to_string(index=False))
