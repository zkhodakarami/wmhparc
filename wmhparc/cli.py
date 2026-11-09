"""Command line interface: ``wmhparc <subcommand> --help``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .atlases import atlas_path, describe, fetch_atlas, get_atlas, list_atlases, derived_dir
from .templates import DEFAULT_TEMPLATE, TEMPLATES

DEFAULT_ATLASES = ["chung20", "cobralab_patho", "bullseye36", "bullseye20"]


def _p(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="auto", choices=["auto", "antspy", "cli"],
                        help="ANTs backend (default: antspy if importable, else ANTs binaries on $PATH)")
    parser.add_argument("--threads", type=int, default=None, help="ITK threads")
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wmhparc", description=__doc__)
    ap.add_argument("--version", action="version", version=f"wmhparc {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list available atlases and templates")

    s = sub.add_parser("info", help="print an atlas' provenance and label table")
    s.add_argument("atlas", choices=list_atlases())

    s = sub.add_parser("fetch", help="download atlases/templates into the cache ($WMHPARC_HOME)")
    s.add_argument("atlases", nargs="*", default=["all"], help="atlas names, 'all', or 'templates'")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("build-bullseye-mni", help="build the 36/20-region bullseye in MNI152NLin6Asym (Harvard-Oxford)")
    s.add_argument("--out", default=None, help="output prefix (default: cache/derived/mni_bullseye)")
    s.add_argument("--insula-to", default="frontal", choices=["frontal", "bgit", "none"])

    s = sub.add_parser("build-bullseye", help="build a subject-space bullseye from FreeSurfer aparc+aseg (NIfTI)")
    s.add_argument("--aparc-aseg", required=True)
    s.add_argument("--out", required=True, help="output prefix")
    s.add_argument("--insula-to", default="frontal", choices=["frontal", "bgit", "none"])
    s.add_argument("--inferior-lateral-ventricles", action="store_true", help="include inf. lat. ventricles in the depth origin")

    s = sub.add_parser("register", help="ANTs SyN registration of T1 (+ rigid FLAIR->T1) or FLAIR-only to a template")
    s.add_argument("--t1", default=None)
    s.add_argument("--flair", default=None)
    s.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    s.add_argument("--out", required=True, help="output directory (gets transforms.json)")
    s.add_argument("--skull-stripped", action="store_true", help="inputs are brain-extracted: register to the template brain")
    s.add_argument("--template-mask", action="store_true", help="restrict the metric to the template brain mask")
    s.add_argument("--quick", action="store_true", help="antsRegistrationSyNQuick-style settings")
    _p(s)

    s = sub.add_parser("parcellate", help="apply atlases to a WMH mask through a transform chain")
    s.add_argument("--wmh", required=True, help="binary WMH mask")
    s.add_argument("--xfm", default=None, help="transforms.json (or its directory) from `register`; omit if the mask is already in MNI")
    s.add_argument("--atlas", nargs="+", default=DEFAULT_ATLASES, help="atlas names or label NIfTI files")
    s.add_argument("--space", default="native", choices=["native", "template"])
    s.add_argument("--wmh-modality", default="FLAIR", choices=["FLAIR", "T1w"], help="which subject image the WMH mask is on")
    s.add_argument("--out", required=True)
    s.add_argument("--prefix", default="sub")
    s.add_argument("--split-masks", action="store_true", help="write one binary mask per region")
    s.add_argument("--fill-mm", type=float, default=None, help="assign unlabelled WMH voxels to the nearest region within MM")
    _p(s)

    s = sub.add_parser("run", help="register + parcellate in one go")
    s.add_argument("--t1", default=None)
    s.add_argument("--flair", default=None)
    s.add_argument("--wmh", required=True)
    s.add_argument("--wmh-modality", default="FLAIR", choices=["FLAIR", "T1w"])
    s.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    s.add_argument("--atlas", nargs="+", default=DEFAULT_ATLASES)
    s.add_argument("--space", default="native", choices=["native", "template"])
    s.add_argument("--out", required=True)
    s.add_argument("--prefix", default="sub")
    s.add_argument("--split-masks", action="store_true")
    s.add_argument("--fill-mm", type=float, default=None)
    s.add_argument("--skull-stripped", action="store_true")
    s.add_argument("--template-mask", action="store_true")
    s.add_argument("--quick", action="store_true")
    _p(s)

    s = sub.add_parser("align-template", help="SyN-register a foreign template (e.g. CoBrALab UKB) to MNI and warp its parcellations")
    s.add_argument("what", help="'cobralab' or a template T1 NIfTI")
    s.add_argument("--labels", nargs="*", default=[], help="label images to warp (when `what` is a file)")
    s.add_argument("--mask", default=None, help="brain mask of the foreign template (when `what` is a file)")
    s.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    s.add_argument("--out", default=None, help="output directory (default: cache/derived/align-<name>)")
    s.add_argument("--quick", action="store_true")
    _p(s)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    quiet = getattr(args, "quiet", False)

    if args.cmd == "list":
        print("Atlases:")
        for n in list_atlases():
            a = get_atlas(n)
            print(f"  {n:18s} [{a.space}] {a.description}")
        print("Templates:")
        for n, t in TEMPLATES.items():
            print(f"  {n:22s} {t.description}")
        return 0

    if args.cmd == "info":
        print(describe(args.atlas))
        return 0

    if args.cmd == "fetch":
        names = args.atlases
        if names == ["all"]:
            names = [n for n in list_atlases() if not get_atlas(n).is_built] + ["templates"]
        for n in names:
            if n == "templates":
                for t in TEMPLATES.values():
                    for k in t.files:
                        t.fetch(k, force=args.force, quiet=quiet)
            elif get_atlas(n).is_built:
                print(atlas_path(n, quiet=quiet))
            else:
                for k, p in fetch_atlas(n, force=args.force, quiet=quiet).items():
                    print(f"{n}:{k} -> {p}")
        return 0

    if args.cmd == "build-bullseye-mni":
        from .bullseye import build_mni_bullseye
        from .imageio import save_labels
        prefix = Path(args.out) if args.out else derived_dir() / "mni_bullseye"
        res = build_mni_bullseye(prefix, quiet=quiet, insula_to=args.insula_to)
        if args.out is None:
            save_labels(res.labels36, res.affine, derived_dir() / "bullseye36_space-MNI152NLin6Asym_dseg.nii.gz")
            save_labels(res.labels20, res.affine, derived_dir() / "bullseye20_space-MNI152NLin6Asym_dseg.nii.gz")
        return 0

    if args.cmd == "build-bullseye":
        from .bullseye import build_from_freesurfer
        build_from_freesurfer(args.aparc_aseg, args.out, quiet=quiet, insula_to=args.insula_to,
                              include_inferior_lateral_ventricles=args.inferior_lateral_ventricles)
        return 0

    if args.cmd == "align-template":
        return _align_template(args, quiet)

    if args.cmd in ("register", "run"):
        from .registration import register_subject
        if args.t1 is None and args.flair is None:
            print("error: give --t1 and/or --flair", file=sys.stderr)
            return 2
        reg_dir = Path(args.out) / "xfm" if args.cmd == "run" else Path(args.out)
        chain = register_subject(t1=args.t1, flair=args.flair, template=args.template, out_dir=reg_dir,
                                 backend=args.backend, skull_stripped=args.skull_stripped, quick=args.quick,
                                 threads=args.threads, use_template_mask=args.template_mask, quiet=quiet)
        if args.cmd == "register":
            return 0
        xfm = chain
    else:
        xfm = args.xfm

    if args.cmd in ("parcellate", "run"):
        from .parcellate import parcellate_many
        parcellate_many(args.wmh, args.atlas, xfm, out_dir=args.out, prefix=args.prefix, space=args.space,
                        wmh_modality=args.wmh_modality, split_masks=args.split_masks, fill_mm=args.fill_mm,
                        backend=args.backend, quiet=quiet)
        return 0

    return 1


def _align_template(args, quiet: bool) -> int:
    from .registration import align_template
    from .atlases import UKB_TEMPLATE, _COBRA_UKB
    if args.what == "cobralab":
        t1 = UKB_TEMPLATE["T1w"].fetch(quiet=quiet)
        mask = UKB_TEMPLATE["mask"].fetch(quiet=quiet)
        labels = [get_atlas(n).resources["labels"].fetch(quiet=quiet) for n in _COBRA_UKB]
        out = Path(args.out) if args.out else derived_dir() / "align-cobralab"
        chain, warped = align_template(t1, args.template, out, labels, template_mask=mask, backend=args.backend,
                                       quick=args.quick, threads=args.threads, quiet=quiet)
        # register the warped files under the names atlas_path() looks for
        import shutil
        for n, w in zip(_COBRA_UKB, warped):
            dst = derived_dir() / f"{n}_space-{args.template}_dseg.nii.gz"
            shutil.copyfile(w, dst)
            print(f"{n} -> {dst}")
    else:
        out = Path(args.out) if args.out else derived_dir() / f"align-{Path(args.what).stem}"
        chain, warped = align_template(args.what, args.template, out, args.labels, template_mask=args.mask,
                                       backend=args.backend, quick=args.quick, threads=args.threads, quiet=quiet)
        for w in warped:
            print(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
