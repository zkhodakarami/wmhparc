import numpy as np

from wmhparc import bullseye as be
from wmhparc.atlases import load_lut


def test_normalized_depth_monotonic(phantom):
    nd = be.normalized_depth(phantom["ventricles"], phantom["cortex"], (1, 1, 1))
    assert nd.min() == 0 and nd.max() <= 1
    assert np.all(nd[phantom["ventricles"]] == 0)
    assert np.allclose(nd[phantom["cortex"]], 1.0)
    # depth increases with radius along the +y axis (away from both ventricles)
    c = int(nd.shape[0] // 2)
    line = nd[c, c:c + 36, c]
    assert np.all(np.diff(line) >= -1e-6)


def test_depth_layers_bins(phantom):
    nd = be.normalized_depth(phantom["ventricles"], phantom["cortex"], (1, 1, 1))
    layers = be.depth_layers(nd, phantom["wm"])
    assert set(np.unique(layers)) == {0, 1, 2, 3, 4}
    assert np.all(layers[~phantom["wm"]] == 0)
    assert np.all(layers[phantom["ventricles"]] == 0)
    assert np.all(layers[phantom["wm"] & (nd < 0.25)] == 1)
    assert np.all(layers[phantom["wm"] & (nd >= 0.75)] == 4)


def test_propagate_labels_respects_hemisphere(phantom):
    lobes = be.propagate_labels(phantom["lobe_seeds"], phantom["wm"], phantom["hemisphere"])
    assert np.all(lobes[phantom["wm"]] > 0)
    left = phantom["wm"] & (phantom["x"] < 0)
    right = phantom["wm"] & (phantom["x"] >= 0)
    assert np.all(lobes[left] // 10 == 1)
    assert np.all(lobes[right] // 10 == 2)


def test_build_bullseye_36_and_20(phantom):
    inp = be.BullseyeInputs(**{k: phantom[k] for k in
                               ("ventricles", "cortex", "wm", "bgt", "lobe_seeds", "hemisphere", "affine")})
    res = be.build_bullseye(inp)
    labs36 = set(np.unique(res.labels36)) - {0}
    # all lobe x layer labels; the small BGIT blob of the phantom does not reach every depth
    assert labs36 >= set(range(1, 37)) - {17, 18, 19, 20}
    assert labs36 & {17, 18, 19, 20}
    labs20 = set(np.unique(res.labels20)) - {0}
    assert labs20 >= set(range(5, 21)) and labs20 & {1, 2, 3, 4}
    # BGIT is labels 17-20 in the 36 scheme and 1-4 in the 20 scheme
    assert set(np.unique(res.labels36[phantom["bgt"]])) <= {17, 18, 19, 20}
    assert set(np.unique(res.labels20[phantom["bgt"]])) <= {1, 2, 3, 4}
    # voxel counts are preserved when folding
    assert (res.labels36 > 0).sum() == (res.labels20 > 0).sum()
    lut36, lut20 = load_lut("bullseye36"), load_lut("bullseye20")
    for _, row in lut36.iterrows():
        n36 = (res.labels36 == row["index"]).sum()
        tgt = lut20[(lut20.region == row.region) & (lut20.layer == row.layer)]["index"].item()
        assert (res.labels20 == tgt)[res.labels36 == row["index"]].all(), row["name"]
        assert n36 <= (res.labels20 == tgt).sum()
    # the periventricular layer is the smallest, the juxtacortical the largest, per region
    for gi in (0, 1, 2, 3, 5, 6, 7, 8):  # lobes (BGIT has no cortex of its own)
        counts = [(res.labels36 == gi * 4 + l).sum() for l in (1, 2, 3, 4)]
        assert counts[0] < counts[-1]


def test_freesurfer_adapter_runs():
    # tiny aparc+aseg-like volume: left/right WM, ventricles, cortex labels, thalamus
    n = 40
    seg = np.zeros((n, n, n), np.int32)
    ii, jj, kk = np.mgrid[:n, :n, :n]
    c = (n - 1) / 2
    r = np.sqrt((ii - c) ** 2 + (jj - c) ** 2 + (kk - c) ** 2)
    left = ii < c
    seg[(r < 16)] = 41; seg[(r < 16) & left] = 2
    seg[(r >= 16) & (r < 18) & left & (jj >= c)] = 1028  # superiorfrontal L
    seg[(r >= 16) & (r < 18) & left & (jj < c)] = 1011   # lateraloccipital L
    seg[(r >= 16) & (r < 18) & ~left & (jj >= c)] = 2028
    seg[(r >= 16) & (r < 18) & ~left & (jj < c)] = 2011
    seg[(r >= 16) & (r < 18) & (np.abs(kk - c) < 2)] = 1002  # cingulate strip (filled from neighbours)
    seg[(r < 3)] = 4
    seg[(r < 3) & ~left] = 43
    seg[(np.sqrt((ii - c + 6) ** 2 + (jj - c) ** 2 + (kk - c) ** 2) < 2)] = 10  # L thalamus
    seg[(r < 16) & left & (np.sqrt((ii - c + 10) ** 2 + (jj - c) ** 2 + (kk - c) ** 2) < 1.5)] = 77  # WM hypo
    inp = be.inputs_from_freesurfer(seg, np.eye(4))
    res = be.build_bullseye(inp)
    assert (res.labels36 > 0).sum() > 0
    present = set(np.unique(res.labels36)) - {0}
    assert any(1 <= l <= 4 for l in present) and any(21 <= l <= 24 for l in present)   # frontal L and R
    assert any(13 <= l <= 16 for l in present) and any(33 <= l <= 36 for l in present)  # occipital L and R
    assert any(17 <= l <= 20 for l in present)                                          # BGIT
    assert np.all(res.lobes[seg == 77] > 0)  # hypointensities got a lobe
