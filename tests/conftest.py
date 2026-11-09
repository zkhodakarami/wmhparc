import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def phantom():
    """Synthetic 'brain': nested spheres on a 1 mm grid.

    ventricle: r < 8 (two blobs, left/right), cortex: 36 <= r < 40, WM in between; a BG blob
    in the left hemisphere.  Lobe seeds split the cortex into 4 lobes by octant.
    """
    n = 96
    c = (n - 1) / 2.0
    ii, jj, kk = np.mgrid[:n, :n, :n]
    x, y, z = ii - c, jj - c, kk - c
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    cortex = (r >= 36) & (r < 40)
    vent = (np.sqrt((x + 8) ** 2 + y ** 2 + z ** 2) < 6) | (np.sqrt((x - 8) ** 2 + y ** 2 + z ** 2) < 6)
    wm = (r < 36) & ~vent
    bgt = np.sqrt((x + 20) ** 2 + (y - 5) ** 2 + z ** 2) < 5
    wm &= ~bgt
    hemisphere = np.where(x < 0, 1, 2).astype(np.int8)
    from wmhparc.bullseye import lobe_code
    seeds = np.zeros((n, n, n), np.int16)
    for h, hm in ((1, x < 0), (2, x >= 0)):
        seeds[cortex & hm & (y >= 0) & (z >= 0)] = lobe_code("frontal", h)
        seeds[cortex & hm & (y >= 0) & (z < 0)] = lobe_code("temporal", h)
        seeds[cortex & hm & (y < 0) & (z >= 0)] = lobe_code("parietal", h)
        seeds[cortex & hm & (y < 0) & (z < 0)] = lobe_code("occipital", h)
    affine = np.eye(4)
    affine[:3, 3] = -c
    return dict(ventricles=vent, cortex=cortex, wm=wm, bgt=bgt, lobe_seeds=seeds, hemisphere=hemisphere,
                affine=affine, r=r, x=x)


def cache_has(relpath: str) -> bool:
    home = Path(os.environ.get("WMHPARC_HOME", Path.home() / ".cache" / "wmhparc"))
    return (home / relpath).exists()


needs_chung = pytest.mark.skipif(not cache_has("atlases/chung20/MNI_ch2better_WM_20ROIs.nii.gz"),
                                 reason="chung20 atlas not in cache (run `wmhparc fetch chung20`)")
needs_ho = pytest.mark.skipif(
    not cache_has("templates/MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_atlas-HOSPA_desc-th0_dseg.nii.gz"),
    reason="Harvard-Oxford atlases not in cache (run `wmhparc fetch templates`)")
needs_cobra = pytest.mark.skipif(not cache_has("atlases/cobralab/WMH_parc_patho_MNI.nii.gz"),
                                 reason="cobralab_patho not in cache")
