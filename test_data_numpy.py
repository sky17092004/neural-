"""
test_data_numpy.py -- test the pure-numpy half of data.py against the real cache.

    python tests/test_data_numpy.py                       # needs cache/
    NW_CACHE=/path/to/cache python tests/test_data_numpy.py

Why the torch stub below exists
------------------------------
`data.py` imports torch only for `WrenchDataset` (tensor conversion). The label
plumbing -- cache loading, the cycle split, normalisation fitting, and the three
target builders that feed evaluate.py's baseline table -- is plain numpy. That
plumbing is exactly where a wrong index silently corrupts every reported number,
so it deserves tests that can run anywhere, including on a machine with no torch
installed. The stub therefore satisfies the import and nothing else; every
function exercised here is the real one, and `WrenchDataset` is deliberately NOT
touched (train.py --smoke covers it once torch is present).
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

if "torch" not in sys.modules:  # pragma: no cover - import shim, see docstring
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.from_numpy = lambda a: a
        stub.tensor = lambda a, **k: a
        stub.initial_seed = lambda: 0
        utils = types.ModuleType("torch.utils")
        utils_data = types.ModuleType("torch.utils.data")
        utils_data.Dataset = object
        utils.data = utils_data
        stub.utils = utils
        sys.modules.update({"torch": stub, "torch.utils": utils,
                            "torch.utils.data": utils_data})

import data as D  # noqa: E402
import pfio  # noqa: E402
import windows as W  # noqa: E402

CACHE = os.environ.get("NW_CACHE", os.path.join(HERE, "..", "cache"))
SPEC = W.WindowSpec(history=3, subframes=4, stride=1)


def _load():
    if not os.path.isfile(os.path.join(CACHE, "meta.json")):
        return None
    return D.load_all_takes(CACHE)


def test_cache_is_self_consistent():
    got = _load()
    if got is None:
        print("   (skipped: no cache -- run preprocess.py first)")
        return
    takes, meta, cams = got
    for tk in takes:
        n = len(tk)
        assert tk.wrench.shape == (n, 6)
        assert tk.pos_mm.shape == (n, 3)
        assert tk.contact.shape == (n,)
        assert tk.T_WT.shape == (n, 4, 4)
        for c in cams:
            assert tk.crops[c].shape == (n, tk.size, tk.size, 3)
            assert tk.uv[c].shape == (n, 2)
        # cycles must partition the take exactly, or the split is not a split
        cover = np.zeros(n, int)
        for s, e in tk.cycles:
            cover[int(s):int(e)] += 1
        assert (cover == 1).all(), f"{tk.name}: cycles do not partition the take"
        assert np.isfinite(tk.wrench).all()


def test_torque_analytic_present_and_physical():
    """The cache must carry r x f, and it must match the measured signature:
    strong on the two lateral axes, blind on the axis along the stick."""
    got = _load()
    if got is None:
        print("   (skipped: no cache)")
        return
    takes, _, _ = got
    tk = takes[0]
    assert tk.torque_analytic is not None, (
        "labels.npz has no 'torque_analytic' -- the cache predates the analytic "
        "baseline, rerun preprocess.py"
    )
    assert tk.torque_analytic.shape == (len(tk), 3)
    strong = np.linalg.norm(tk.wrench[:, :3], axis=1) > 20
    if strong.sum() < 10:
        return
    corr = [float(np.corrcoef(tk.torque_analytic[strong, i],
                             tk.wrench[strong, 3 + i])[0, 1]) for i in range(3)]
    lateral = [c for i, c in enumerate(corr) if i != pfio.PUSH_AXIS]
    assert all(c > 0.8 for c in lateral), f"lateral torque corr {lateral}"
    # r = arm * z_tool is parallel to the stick, so r x f has no component along
    # the stick: the twist channel is structurally unexplainable this way.
    assert np.abs(tk.torque_analytic[strong, pfio.PUSH_AXIS]).max() < 0.15


def test_split_windows_disjoint_on_real_cache():
    got = _load()
    if got is None:
        print("   (skipped: no cache)")
        return
    takes, _, _ = got
    train, val, info = D.split_windows(takes, SPEC, 0.25)
    assert train and val, info
    for ti, tk in enumerate(takes):
        tr = {int(i) for w in train if w.take == ti
              for i in w.target_idx[w.target_valid]}
        va = {int(i) for w in val if w.take == ti
              for i in w.target_idx[w.target_valid]}
        assert not (tr & va), f"{tk.name}: {len(tr & va)} frames in both splits"
        tr_in = {int(i) for w in train if w.take == ti for i in w.input_idx}
        assert not (tr_in & va), f"{tk.name}: training inputs see val frames"


def test_norm_uses_training_frames_only():
    """A norm fitted on train must differ from one fitted on everything, and must
    reproduce the mean of exactly the training frames."""
    got = _load()
    if got is None:
        print("   (skipped: no cache)")
        return
    takes, _, _ = got
    train, val, _ = D.split_windows(takes, SPEC, 0.25)
    norm = D.fit_norm(takes, train)
    frames = {(w.take, int(i)) for w in train for i in w.target_idx[w.target_valid]}
    ref = np.array([takes[t].wrench[i] for t, i in sorted(frames)])
    assert np.allclose(norm.wrench_mean, ref.mean(0)), "norm saw the wrong frames"
    all_mean = np.concatenate([t.wrench for t in takes]).mean(0)
    assert not np.allclose(norm.wrench_mean, all_mean, atol=1e-6), (
        "train-only norm equals the all-frames norm; the split is not holding "
        "anything out"
    )
    nw = norm.norm_wrench(ref)
    assert np.allclose(nw.std(0), 1.0, atol=0.05)


def test_target_builders_agree_and_align():
    got = _load()
    if got is None:
        print("   (skipped: no cache)")
        return
    takes, _, _ = got
    train, val, _ = D.split_windows(takes, SPEC, 0.25)
    wins = train + val
    gt, valid, contact = D.target_wrenches(takes, wins)
    anch = D.anchor_wrenches(takes, wins)
    ta = D.analytic_torque_targets(takes, wins)
    M, K = len(wins), SPEC.subframes
    assert gt.shape == (M, K, 6) and valid.shape == (M, K) and contact.shape == (M,)
    assert anch.shape == (M, 6)
    assert ta is not None and ta.shape == (M, K, 3), "analytic torque row is missing"
    # offset 0 is the observed frame, so the anchor must equal the offset-0 target
    assert np.allclose(gt[:, 0, :], anch), "anchor and offset-0 target disagree"
    assert valid[:, 0].all()
    # every builder must index the same frames in the same order
    for i, w in enumerate(wins[::7]):
        j = i * 7
        tk = takes[w.take]
        assert np.allclose(ta[j, 0], tk.torque_analytic[w.anchor])
        assert bool(contact[j]) == bool(tk.contact[w.anchor])


def test_hybrid_row_is_constructible():
    """Reproduce evaluate.py's substitution on ground truth: swapping the true
    torques for r x f must leave the force channels untouched and change only
    the torque channels."""
    got = _load()
    if got is None:
        print("   (skipped: no cache)")
        return
    takes, _, _ = got
    train, val, _ = D.split_windows(takes, SPEC, 0.25)
    wins = train + val
    gt, _, _ = D.target_wrenches(takes, wins)
    ta = D.analytic_torque_targets(takes, wins)
    hybrid = gt.copy()
    hybrid[:, :, 3:] = ta
    assert np.allclose(hybrid[:, :, :3], gt[:, :, :3])
    assert not np.allclose(hybrid[:, :, 3:], gt[:, :, 3:])
    err = np.abs(hybrid[:, :, 3:] - gt[:, :, 3:]).mean()
    assert 0.0 < err < 1.0, f"analytic torque is off by {err:.3f} Nm on average"


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
