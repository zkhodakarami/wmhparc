"""Tests that need the downloaded atlases (skipped when the cache is empty)."""
import numpy as np

from wmhparc.atlases import load_atlas, load_lut, list_atlases
from tests.conftest import needs_chung, needs_cobra, needs_ho


def test_luts_load_for_every_atlas():
    for name in list_atlases():
        lut = load_lut(name)
        assert {"index", "name"} <= set(lut.columns)
        assert lut["index"].is_unique and (lut["index"] > 0).all()


@needs_chung
def test_chung20_structure_matches_lut():
    """The inferred Chung label table: 5 chains of 4 adjacent labels, layer 1 smallest & mutually
    adjacent (periventricular), layer 4 largest (juxtacortical)."""
    img, lut = load_atlas("chung20")
    d = np.asarray(img.dataobj)[::2, ::2, ::2]
    assert set(np.unique(d)) == set(range(21))
    adj = np.zeros((21, 21), int)
    for ax in range(3):
        a = np.take(d, range(0, d.shape[ax] - 1), axis=ax)
        b = np.take(d, range(1, d.shape[ax]), axis=ax)
        m = a != b
        np.add.at(adj, (a[m], b[m]), 1)
        np.add.at(adj, (b[m], a[m]), 1)
    counts = np.bincount(d.ravel(), minlength=21)
    for g in range(5):
        l1, l2, l3, l4 = (g * 4 + k for k in (1, 2, 3, 4))
        assert adj[l1, l2] > 0 and adj[l2, l3] > 0 and adj[l3, l4] > 0
        assert adj[l1, l3] == 0 and adj[l1, l4] == 0 and adj[l2, l4] == 0, "layers must be concentric"
        if g > 0:  # lobes: juxtacortical layer is the largest; BGIT layer 4 is tiny (lies against the insula)
            assert counts[l4] > counts[l1]
    layer1 = [1, 5, 9, 13, 17]
    for a in layer1:
        assert sum(adj[a, b] > 0 for b in layer1 if b != a) >= 2
    # region identity (bounding boxes in mm): frontal most anterior, occipital most posterior, temporal lowest
    aff = img.affine.copy(); aff[:3, :3] *= 2
    def bbox(label):
        idx = np.argwhere(d == label)
        return (aff @ np.r_[idx.min(0), 1])[:3], (aff @ np.r_[idx.max(0), 1])[:3]
    assert bbox(8)[1][1] > 60           # frontal_layer4 reaches y > 60 mm
    assert bbox(12)[0][1] < -95         # occipital_layer4 reaches y < -95 mm
    assert bbox(16)[0][2] < -40         # temporal_layer4 reaches z < -40 mm
    assert bbox(20)[1][2] > 75          # parietal_layer4 reaches z > 75 mm
    assert np.all(np.abs(bbox(1)[0]) < 45) and np.all(np.abs(bbox(1)[1]) < 45)  # BGIT central


@needs_cobra
def test_cobralab_patho_labels_and_geometry():
    img, lut = load_atlas("cobralab_patho")
    d = np.asarray(img.dataobj)
    assert set(np.unique(d)) == {0, 1, 2, 3}
    assert img.shape == (193, 229, 193)
    cy = {l: (img.affine @ np.r_[np.argwhere(d == l).mean(0), 1])[1] for l in (1, 2, 3)}
    assert cy[2] < cy[1] < cy[3]  # posterior < periventricular < anterior along y


@needs_ho
def test_mni_bullseye_builds(tmp_path):
    from wmhparc.bullseye import build_mni_bullseye
    res = build_mni_bullseye(None, quiet=True)
    assert set(np.unique(res.labels36)) == set(range(37))
    assert set(np.unique(res.labels20)) == set(range(21))
    # left-hemisphere regions have negative mean x
    for l in (1, 5, 9, 13):
        idx = np.argwhere(res.labels36 == l)
        assert (res.affine @ np.r_[idx.mean(0), 1])[0] < 0
    assert ((res.lobes == 0) & (res.layers > 0)).sum() == 0
