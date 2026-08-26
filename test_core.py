"""
test_core.py -- tests for the torch-free core (pfio, windows, metrics).

    python tests/test_core.py          # standalone, no pytest needed
    pytest tests/test_core.py -q       # also works

These cover the parts where a silent bug is most expensive: the projection
convention, the unit conversions, the window grid, and above all the train/val
split, because a leaky split produces excellent numbers that mean nothing.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import metrics  # noqa: E402
import pfio  # noqa: E402
import windows as W  # noqa: E402

DATA_ROOT = os.environ.get("POKEFLEX_ROOT", "")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def test_projection_identity():
    """With T = identity, uv must equal the pinhole projection of the point."""
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    cam = pfio.Camera("c0", "kinect", 0, K, np.zeros(5), _static_extr=np.eye(4))
    p = np.array([0.1, -0.05, 2.0])
    uv, z = cam.project(p, 1)
    assert np.isclose(z, 2.0)
    assert np.allclose(uv, [320 + 600 * 0.05, 240 - 600 * 0.025])


def test_projection_translation_and_batch():
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 100.0
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.0]  # camera 1 m behind the world origin
    cam = pfio.Camera("c0", "volucam", 0, K, np.zeros(5), _static_extr=T)
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    uv, z = cam.project(pts, 7)
    assert uv.shape == (2, 2) and z.shape == (2,)
    assert np.allclose(z, [1.0, 2.0])
    assert np.allclose(uv[:, 0], 0.0)  # on the optical axis


def test_to_camera_preserves_units_and_shape():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    cam = pfio.Camera("c", "kinect", 0, np.eye(3), np.zeros(5), _static_extr=T)
    single = cam.to_camera(np.zeros(3), 1)
    assert single.shape == (3,) and np.allclose(single, [1, 2, 3])
    batch = cam.to_camera(np.zeros((4, 3)), 1)
    assert batch.shape == (4, 3)


def test_camera_centre_round_trip():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = [0.3, -0.2, 1.5]
    cam = pfio.Camera("c", "kinect", 0, np.eye(3), np.zeros(5), _static_extr=T)
    centre = cam.centre_world(1)
    # the camera centre must map to the camera-frame origin
    assert np.allclose(cam.to_camera(centre, 1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# contact runs and cycles
# --------------------------------------------------------------------------- #


def test_contact_runs():
    m = np.array([0, 1, 1, 0, 0, 1, 0], bool)
    assert pfio.contact_runs(m) == [(1, 3), (5, 6)]
    assert pfio.contact_runs(np.ones(3, bool)) == [(0, 3)]
    assert pfio.contact_runs(np.zeros(3, bool)) == []


def test_poke_cycles_tile_exactly():
    m = np.zeros(40, bool)
    m[5:9] = m[18:22] = m[31:35] = True
    cyc = pfio.poke_cycles(m)
    assert len(cyc) == 3
    assert cyc[0][0] == 0 and cyc[-1][1] == 40
    for (_, a), (b, _) in zip(cyc[:-1], cyc[1:]):
        assert a == b, "cycles must be contiguous"
    covered = np.zeros(40, int)
    for s, e in cyc:
        covered[s:e] += 1
    assert (covered == 1).all(), "cycles must partition the take"


def test_split_cycles_disjoint():
    # ceil(0.25 * 5) = 2 cycles held out
    tr, va = pfio.split_cycles(5, None, 0.25)
    assert va == [3, 4] and tr == [0, 1, 2]
    assert not set(tr) & set(va)
    assert sorted(tr + va) == list(range(5)), "every cycle must be assigned"
    tr, va = pfio.split_cycles(8, None, 0.25)
    assert va == [6, 7] and len(tr) == 6
    # explicit override
    tr, va = pfio.split_cycles(5, [1, 3], 0.25)
    assert va == [1, 3] and tr == [0, 2, 4]
    # a single cycle cannot be split, so nothing is held out
    tr, va = pfio.split_cycles(1, None, 0.25)
    assert tr == [0] and va == []


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #


def test_window_shapes_and_ordering():
    spec = W.WindowSpec(history=5, subframes=4)
    wins = W.build_windows(100, [(0, 100)], spec)
    assert wins, "expected at least one window"
    for w in wins:
        assert len(w.input_idx) == 5 and len(w.target_idx) == 4
        assert w.input_idx[-1] == w.anchor, "newest input must be the anchor"
        assert (np.diff(w.input_idx) >= 0).all(), "inputs must be ordered oldest->newest"
        assert (w.input_idx <= w.anchor).all(), "no input may come from the future"
        assert w.target_idx[0] == w.anchor
        assert w.target_valid[0]


def test_window_targets_stay_in_block():
    spec = W.WindowSpec(history=3, subframes=4)
    wins = W.build_windows(50, [(10, 26)], spec)
    for w in wins:
        assert 10 <= w.anchor < 26
        assert (w.input_idx >= 10).all(), "history must not cross the block start"
        assert (w.target_idx[w.target_valid] < 26).all()
    # the last anchor is padded, so some targets must be invalid somewhere
    assert any(not w.target_valid.all() for w in wins)


def test_no_frame_leaks_between_splits():
    """The decisive test. Train and val targets must be disjoint frame sets."""
    mask = np.zeros(160, bool)
    for s in (10, 42, 74, 106, 138):
        mask[s : s + 12] = True
    cyc = pfio.poke_cycles(mask)
    tr_i, va_i = pfio.split_cycles(len(cyc))
    spec = W.WindowSpec(history=5, subframes=4)
    tr = W.build_windows(160, [cyc[i] for i in tr_i], spec)
    va = W.build_windows(160, [cyc[i] for i in va_i], spec)
    tr_f = {int(i) for w in tr for i in w.target_idx[w.target_valid]}
    va_f = {int(i) for w in va for i in w.target_idx[w.target_valid]}
    assert tr_f and va_f
    assert not (tr_f & va_f), f"{len(tr_f & va_f)} frames appear in both splits"
    # inputs may not reach into the other split either
    tr_in = {int(i) for w in tr for i in w.input_idx}
    assert not (tr_in & va_f), "training inputs must not touch validation frames"


def test_anchor_grid_is_global():
    """Anchors from different blocks must lie on one global grid of period step,
    otherwise train and val windows can cover the same interval."""
    spec = W.WindowSpec(history=2, subframes=4)
    a1 = [w.anchor for w in W.build_windows(200, [(0, 60)], spec)]
    a2 = [w.anchor for w in W.build_windows(200, [(61, 130)], spec)]
    for a in a1 + a2:
        assert a % 4 == 0


def test_dense_protocol_covers_every_frame():
    spec = W.WindowSpec(history=4, subframes=1)
    wins = W.build_windows(30, [(0, 30)], spec)
    assert len(wins) == 30
    assert {w.anchor for w in wins} == set(range(30))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #


def test_zero_order_hold():
    a = np.arange(12, dtype=float).reshape(2, 6)
    out = W.zero_order_hold(a, 4)
    assert out.shape == (2, 4, 6)
    for k in range(4):
        assert np.allclose(out[:, k], a)


def test_linear_interp_endpoints():
    a = np.zeros((3, 6))
    a[1] = 1.0
    a[2] = 2.0
    out = W.linear_interp(a, 4)
    assert out.shape == (3, 4, 6)
    assert np.allclose(out[:, 0], a), "offset 0 must reproduce the anchor exactly"
    assert np.allclose(out[0, :, 0], [0, 0.25, 0.5, 0.75])
    assert np.allclose(out[2, :, 0], 2.0), "last anchor falls back to a hold"


def test_linear_beats_hold_on_a_ramp():
    t = np.linspace(0, 10, 40)
    dense = np.stack([t] * 6, 1)  # a perfect ramp at the dense rate
    anchors = dense[::4]
    gt = np.stack([dense[i * 4 : i * 4 + 4] for i in range(len(anchors) - 1)])
    zoh = W.zero_order_hold(anchors[:-1], 4)
    lin = W.linear_interp(anchors, 4)[:-1]
    e_zoh = np.abs(zoh - gt).mean()
    e_lin = np.abs(lin - gt).mean()
    assert e_lin < 1e-9 < e_zoh, (e_lin, e_zoh)


def test_frame_time():
    assert W.frame_time(1) == 0.0
    assert np.isclose(W.frame_time(31), 1.0)


# --------------------------------------------------------------------------- #
# normalisation and metrics
# --------------------------------------------------------------------------- #


def test_norm_round_trip_and_per_axis():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(200, 6)) * np.array([30, 40, 10, 0.3, 0.2, 0.5])
    p = rng.normal(size=(200, 3)) * 50 + 400
    ns = pfio.NormStats.fit(w, p)
    assert np.allclose(ns.denorm_wrench(ns.norm_wrench(w)), w)
    assert np.allclose(ns.denorm_pos(ns.norm_pos(p)), p)
    nw = ns.norm_wrench(w)
    assert np.allclose(nw.std(0), 1.0, atol=0.05), "each axis must be unit variance"
    assert np.allclose(nw.mean(0), 0.0, atol=1e-9)


def test_norm_save_load(tmp="/tmp/_nw_norm.json"):
    ns = pfio.NormStats([1, 2, 3, 4, 5, 6], [1] * 6, [0, 0, 0], [1, 1, 1])
    ns.save(tmp)
    back = pfio.NormStats.load(tmp)
    assert back.wrench_mean == ns.wrench_mean
    os.remove(tmp)


def test_metrics_perfect_and_mean():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(20, 4, 6))
    valid = np.ones((20, 4), bool)
    s = metrics.wrench_scores(gt, gt, valid)
    assert np.allclose(s["rmse"], 0) and np.allclose(s["mae"], 0)
    assert np.allclose(s["r2"], 1.0)
    mean_pred = np.broadcast_to(gt.reshape(-1, 6).mean(0), gt.shape)
    s2 = metrics.wrench_scores(mean_pred, gt, valid)
    assert np.all(np.abs(s2["r2"]) < 1e-9), "predicting the mean must give R2 = 0"


def test_evaluate_groups():
    gt = np.zeros((10, 4, 6))
    pred = np.ones((10, 4, 6))
    valid = np.ones((10, 4), bool)
    valid[:, 3] = False
    contact = np.zeros(10, bool)
    contact[:4] = True
    sc = metrics.evaluate_wrench(pred, gt, valid, contact)
    assert sc["observed"]["n"] == 10
    assert sc["subframe"]["n"] == 20  # offsets 1 and 2 only, offset 3 is invalid
    assert sc["contact"]["n"] == 12
    assert sc["free"]["n"] == 18
    assert np.allclose(sc["all"]["rmse"], 1.0)


def test_position_scores():
    gt = np.zeros((5, 3))
    pred = np.zeros((5, 3))
    pred[:, 0] = 3.0
    pred[:, 1] = 4.0
    s = metrics.position_scores(pred, gt)
    assert np.isclose(s["mean_mm"], 5.0)


# --------------------------------------------------------------------------- #
# real data (skipped unless POKEFLEX_ROOT is set)
# --------------------------------------------------------------------------- #


def test_real_take_conventions():
    if not DATA_ROOT or not os.path.isdir(DATA_ROOT):
        print("   (skipped: set POKEFLEX_ROOT to run the real-data checks)")
        return
    takes = pfio.find_takes(DATA_ROOT)
    assert takes, f"no labelled takes under {DATA_ROOT}"
    log = pfio.load_robot_log(takes[0])
    assert log.wrench.shape[1] == 6
    assert np.array_equal(log.fid, np.sort(log.fid))

    # the tool frame is rigidly attached to the flange: 208 mm along tool +z.
    # The logged matrices are rounded, which shows up as ~1.4 mm of jitter on the
    # z offset, so the tolerance is 3 mm rather than micrometres.
    rel = np.einsum("nij,njk->nik", np.linalg.inv(log.T_WE), log.T_WT)
    off_mm = rel[:, :3, 3] * 1000
    assert 150 < off_mm[:, 2].mean() < 260, off_mm[:, 2].mean()
    assert np.abs(off_mm[:, :2]).max() < 3.0, "the stick must be along tool z"
    assert off_mm[:, 2].std() < 3.0, "the stick length must be constant"

    # `forces` is in WORLD axes, not the tool frame: the logged force direction
    # must oppose the stick axis (-z_tool expressed in world) with no rotation.
    c = pfio.contact_mask(log)
    strong = c & (np.linalg.norm(log.wrench[:, :3], axis=1) > 20)
    assert strong.sum() > 5, "not enough loaded frames to test the frame"
    f = log.wrench[strong, :3]
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    axis = -log.T_WT[strong, :3, 2]
    cos_world = float(np.mean(np.einsum("ni,ni->n", f, axis)))
    R = log.T_WT[strong, :3, :3]
    f_rot = np.einsum("nij,nj->ni", R, f)
    cos_tool = float(np.mean(np.einsum("ni,ni->n", f_rot, axis)))
    assert cos_world > 0.9, f"world-frame hypothesis failed (cos={cos_world:.3f})"
    assert cos_tool < 0.5, f"tool-frame hypothesis should fail (cos={cos_tool:.3f})"

    # the logged wrench must be dominated by the pushing axis
    dom = int(np.argmax(np.abs(log.wrench[:, :3]).mean(0)))
    assert dom == pfio.PUSH_AXIS

    # the analytic r x f torque must explain the two lateral torque channels
    ta = log.analytic_torque()
    for i in (0, 2):
        r = float(np.corrcoef(ta[strong, i], log.wrench[strong, 3 + i])[0, 1])
        assert r > 0.8, f"analytic torque axis {i} correlation only {r:.2f}"

    # T_camera<-world convention: the tip must project inside the eye-in-hand view
    from PIL import Image
    cam = pfio.load_camera(takes[0], "realsense0")
    Wd, Ht = Image.open(pfio.image_path(takes[0], "realsense0", int(log.fid[0]))).size
    inside = 0
    for k in range(len(log)):
        uv, z = cam.project(log.T_WT[k, :3, 3], int(log.fid[k]))
        inside += z > 0 and 0 <= uv[0] < Wd and 0 <= uv[1] < Ht
    frac = inside / len(log)
    assert frac > 0.95, f"only {frac:.1%} of tip projections land in the image"
    print(f"   (real data: {pfio.take_name(takes[0])}, {len(log)} frames, "
          f"{frac:.1%} projections in image, cos_world={cos_world:.3f})")


# --------------------------------------------------------------------------- #


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
