# neural_wrench — forces and contact position from PokeFlex images

Given the camera images of a poke, recover the **six-component wrench**
(`Fx Fy Fz Tx Ty Tz`) and the **3D contact position**, for *every* image — and,
because the force/torque sensor logs at 120 Hz while the cameras run at 30 Hz,
recover **four wrenches per image** rather than one.

This is the *inverse* of what PokeFlex's own released code does. They map
`force + contact position → deformed mesh`. Here we map
`images → force + contact position`.

---

## 1. Start here

This repo is installed at `karan/neural_wrench/`, right next to your two data
folders, so the data root is simply the parent directory, `..`:

```
karan/
├── octupus_t1/        <- labelled take
├── octopus_t2/        <- unlabelled take
└── neural_wrench/     <- this repo   ... run everything from in here
```

```bash
cd neural_wrench
pip install -r requirements.txt

# fast plumbing check: ~2 minutes, CPU only, nothing to download
SMOKE=1 bash scripts/run_octopus.sh ..

# the real thing
bash scripts/run_octopus.sh ..
```

`--data_root` is always the folder that **contains** `octupus_t1/`, never
`octupus_t1` itself. If you move the repo elsewhere, pass that folder instead
of `..`.

Three test suites. The last two need no GPU, and `test_static.py` needs nothing
installed at all:

```bash
POKEFLEX_ROOT=.. python3 tests/test_core.py   # 22 tests, reads the real take
python3 tests/test_data_numpy.py              # 6 tests, needs cache/ to exist
python3 tests/test_static.py                  # 6 checks, no dependencies
```

---

## 2. What is in the downloaded data, and what actually matters

Measured on the two takes you have (5.7 GB + 2.0 GB). **Read the last column
first** — most of those gigabytes are not useful for this task.

| Path in a take | Count | Size | What it is | Use it? |
|---|---|---|---|---|
| `robot_data.json` | 1 | 0.2 MB | **The labels.** One entry per image frame: `forces` (6), `T_WT`, `T_WE`, `T_WB`, `jointPos` | **Essential.** Without this a take cannot be trained on |
| `realsense/{0,1}/color/*.png` | 155 each | ~85 MB | 848×480 **eye-in-hand** views, bolted to the robot arm | **Primary input.** Close to the contact, so deformation fills the frame |
| `realsense/{0,1}/camera_parameters.json` | 1 each | tiny | Intrinsics + **per-frame** extrinsics (these cameras move) | **Essential** for projecting the tip into the image |
| `realsense/{0,1}/depth/*.png` | 155 each | ~30 MB | Depth, 10000 units per metre | Optional, `--with_depth` |
| `kinect/{0,1}/color/*.png` | 155 each | ~1.4 GB | 3840×2160 static wide views. 8.7 MB per frame | Optional. Costly to decode, distant from the contact |
| `kinect/{0,1}/depth/*.png` | 155 each | ~24 MB | Depth, 1000 units per metre | Optional |
| `volucam/{0,1}/color/*.png` | 155 each | ~280 MB | Two more static views (t1 only) | Optional extra viewpoints |
| `meshes/mesh-fXXXXX.obj` | 155 | 2.1 GB | Reconstructed **deformed surface**, 21 297 verts, 40 000 faces, millimetres, same world frame as `T_WT × 1000` | **Not needed here.** This is the *output* of PokeFlex's task and the input to nothing of ours |
| `meshes/Atlas-FXXXXX.png` + `.mtl` | 310 | — | Texture atlases for those meshes | Not needed |
| `mesh_confidence/*.npy` | 155 | 26 MB | float64 `(21297,)` in `[0,1]` — **per-vertex** reconstruction confidence | Not needed |
| `System Volume Information/` | — | — | Windows artefact from the external drive | Ignore (the loader skips it) |

**So: of 7.7 GB, the pipeline needs about 200 MB** — two RealSense colour
streams, their camera parameters, and `robot_data.json`.

### The two takes are not equivalent

- **`octupus_t1`** — 155 frames, **has `robot_data.json`**, has meshes, has
  volucams. This is the only take you can train or score on.
- **`octopus_t2`** — 155 frames, **no `robot_data.json`**. No labels, so it can
  only be used for prediction, never for evaluation. `kinect/0/color` has just 38
  of 155 files, so that stream is a partial download. `find_takes()` skips t2
  automatically; `find_unlabelled_takes()` reports it.

### The honest size problem

`octupus_t1` is **155 frames, 5 pokes, and only 5 distinct contact sites** — the
stick tip steps up the octopus in five stages, at world z ≈ **115, 144, 177, 210,
244 mm**, each held for one poke. The position head therefore has five distinct
labels to learn from. This is enough to prove the pipeline runs and to produce a
defensible methodology chapter; it is **not** enough to train a model that
generalises. `preprocess.py` prints a warning below 2000 frames. Download more
takes and more objects from <https://pokeflex.ait.ethz.ch/> before you report a
headline number.

---

## 3. What each file in this repo does

Read them in this order — it is also the order data flows through them.

| File | Role |
|---|---|
| `src/pfio.py` | **Read this first.** All I/O and every geometric convention: the `robot_data.json` schema, the camera model, unit conversions, poke-cycle segmentation, per-axis normalisation. Torch-free and heavily tested, because a wrong extrinsic convention produces a model that trains happily and means nothing. The module docstring records what was *measured*, not assumed. |
| `src/probe.py` | Inspects raw takes and prints the schema, force ranges, and the frame-to-frame force jumps. Run it before anything else on a new take. |
| `src/windows.py` | The interleave protocol: builds `(input frames, target sub-frames)` windows on a **global** anchor grid, plus the `zero_order_hold` / `linear_interp` baselines. |
| `src/preprocess.py` | Raw PNGs → a 224×224 uint8 crop cache plus `labels.npz`. One take is ~23 MB per camera and loads instantly, versus 1.4 GB of Kinect PNGs decoded every epoch. |
| `src/data.py` | Dataset over the cache. Owns the **split by poke cycle**, the train-only normalisation fit, and the target builders that feed the baseline table. |
| `src/model.py` | `WrenchNet`: per-frame backbone → transformer over L×C tokens → four heads. `python src/model.py --smoke` self-tests it. |
| `src/losses.py` | Masked Huber on the wrench, contact-masked position loss, contact BCE, uv aux loss, temporal smoothness. |
| `src/train.py` | The training loop. Writes `config.json`, `norm.json`, `best.pt`, `log.jsonl`. `--smoke` runs the whole thing in ~2 min on a CPU. |
| `src/evaluate.py` | Scores a run against every reference predictor and writes one CSV row per image × sub-frame. |
| `src/infer.py` | Label-free prediction on a raw take. This is what you run on `octopus_t2`. |
| `src/viz.py` | Plots. Torch-free, so it works on a machine with no GPU. |
| `src/metrics.py` | RMSE / MAE / R² per axis, grouped by observed vs interpolated and contact vs free. |
| `tests/test_core.py` | 22 tests on the torch-free core, including the decisive leakage test and a real-data convention check. |
| `tests/test_data_numpy.py` | 6 tests on the label plumbing, runnable with **no torch installed**. |
| `tests/test_static.py` | 6 whole-repo consistency checks with **no dependencies at all**: every file compiles, every cross-file reference resolves (both `pfio.foo()` and `from pfio import foo`), and every `--flag` *and its value* used in the pipeline script or this README is accepted by the script it is passed to. Each check was verified by deliberately breaking the repo and confirming it fails. Two real bugs already caught: `infer.py` called with `--out` when it declares `--csv`, and `--crop_mode center` when the valid choices are `full/fixed/fk_centered`. |
| `scripts/run_octopus.sh` | The whole pipeline, and the de-facto config file. |

---

## 4. The 30 Hz / 120 Hz problem, and what this repo honestly does about it

Your framing was right, and the numbers back it up. On `octupus_t1`:

- total `Fy` range **90.6 N** over a 5.13 s take (155 frames at 30 fps)
- mean force change between two consecutive images, during contact: **7.4 N**
- **largest** change between two consecutive images: **32.4 N** (≈ 973 N/s)

One 30 Hz sample genuinely hides tens of newtons of variation, so predicting a
single wrench per image is throwing away most of the signal.

**But there is a catch you need to know about.** The `robot_data.json` that ships
with the public release is **already collapsed to one entry per image frame**.
The four intermediate 120 Hz samples are not in the data you have. No amount of
code recovers them.

So the repo does the next best thing, which is a real experiment rather than a
fake one — the **interleave protocol**:

- train on every K-th frame (K=4 → an effective **7.5 Hz** camera),
- ask the model for **K wrenches** per input,
- score against the **held-out intermediate frames**, which have true labels.

That is a faithful proxy for 30 → 120 Hz with honest ground truth, and it answers
exactly the question that matters: *can a model infer sub-frame force history from
images?* If you later obtain the raw 120 Hz log, keep `--subframes 4` and point
`preprocess.py` at the dense log — nothing else in the pipeline changes.

### How to read the results table

`evaluate.py` prints these rows. The comparison is the point, not the absolute
number.

| Row | Meaning |
|---|---|
| `model` | the network's K sub-frame predictions |
| `model_zoh` | the network's own offset-0 output, repeated K times |
| `model_F + rxf_torque` | the model's forces, with torques replaced by `r × f` from the true pose |
| `oracle_zoh` | the **true** wrench at the observed frame, repeated K times |
| `oracle_linear` | linear interpolation between **true** observed wrenches |
| `train_mean` | the training-set mean |

- **`model` vs `model_zoh`** is the honest headline: does predicting a sub-frame
  *trajectory* beat repeating a single estimate? If these two rows are equal, the
  K-way head is decorative.
- **`model` vs `train_mean`**: has it learnt anything at all?
- **`oracle_*`** rows are given ground truth the model never sees. They are an
  upper reference, **not** a competitor. Do not report "we beat the oracle".
- **`model_F + rxf_torque`** — see below.

---

## 5. Five findings from the data you should not re-derive

All measured on `octupus_t1`, all recorded in the source docstrings.

1. **The extrinsics are `T_camera←world`, OpenCV convention.** Verified visually,
   not assumed: projecting the stick axis from `T_WE[:3,3]` to `T_WT[:3,3]` traces
   the visible acrylic rod exactly, and the tip lands where the rod enters the
   plush, in all views. Contact-position labels are therefore trustworthy, and 2D
   contact labels come for free.

2. **`forces` is in the WORLD frame, not the tool frame.** Measured over the 39
   frames above 20 N, the cosine between the logged force direction and the
   expected reaction direction is **+0.988** as-logged, versus +0.068 rotated by
   `R_WB` and +0.055 by `R_WT`. (`R_WB` is not identity, so the test discriminates.)
   Rotating into the tool frame moves all the load onto `Fz`, spanning
   −91.5 → −0.8 N — clean axial compression, as physics demands. Hence
   `--wrench_frame tool` exists and is the better-posed target for eye-in-hand
   input.

3. **The torques are mostly redundant.** A single lever arm of **105 mm** from the
   tip along tool `+z` reproduces `Tx` at correlation **0.934** and `Tz` at
   **0.950**, residual ≈ 0.18 Nm. This is why the `model_F + rxf_torque` row
   exists: **a learned torque head has to beat `r × f` to be worth its weight.**

4. **One torque channel is structurally unexplainable that way.** With
   `r ∥ z_tool`, the cross product `r × f` has *no component along the stick*, so
   it cannot produce twist about the stick axis. That is exactly the channel it
   fails on (`Ty`, correlation 0.16, a constant +0.427 Nm tare). If you want that
   channel, it must come from the images or from the sensor — not from geometry.

5. **Per-axis normalisation is mandatory.** Forces reach 91 N while torques stay
   under 1.6 Nm — a **59×** magnitude ratio. PokeFlex divides everything by 100;
   with a shared scale the torque residuals are ~60× smaller than the force
   residuals and the network simply stops fitting them.

---

## 6. Three ways to accidentally fake your own results

Each of these is guarded in code, and each guard defaults to the honest setting.

1. **Splitting by frame instead of by poke.** At 30 fps, frame *t* and *t+1* are
   nearly the same image with nearly the same wrench, so a random frame split
   turns validation into a nearest-neighbour lookup into the training set. The
   split is therefore by **poke cycle**, and `test_no_frame_leaks_between_splits`
   asserts the two frame sets are disjoint — inputs included.

2. **`--crop_mode fk_centered`.** Centring the crop on the forward-kinematics tip
   projection is legitimate for the *wrench* head (FK is available online on a
   real robot), but it places the answer to the *position* head at the crop centre
   by construction. `train.py` refuses to weight the position head in this mode
   unless you pass `--allow_fk_position`.

3. **`--use_pose`.** Feeding the FK tool pose hands over the position label
   directly. `train.py` disables the position loss and tells you to report wrench
   only.

Also: augmentation is **photometric only** by default. Geometric jitter moves the
object in the image without changing the 3D position label, which teaches the
position head that one pixel pattern maps to several answers. `--aug_shift` exists
for wrench-only runs.

---

## 7. Outputs

`evaluate.py --csv` and `infer.py --csv` both write one row per image × sub-frame
— the per-image answer you asked for:

```
take, frame, offset, observed, Fx_pred..Tz_pred, Fx_gt..Tz_gt,
x_pred_mm, y_pred_mm, z_pred_mm, x_gt_mm, y_gt_mm, z_gt_mm,
contact_prob, contact_gt
```

`offset` is 0…K−1 within the image, and `observed = 1` marks the sub-frame that
coincides with the camera exposure. `infer.py` omits the `_gt` columns, because on
an unlabelled take there is nothing to compare against.

---

## 8. Honest limitations

- 155 labelled frames, 5 pokes, **5 distinct contact sites**, one object, one
  approach direction. Treat every number as a pipeline check, not a result.
- The 120 Hz ground truth is not in the public release; the interleave protocol
  is a proxy, and the report says so.
- The 105 mm moment arm is fitted on this one take. Re-fit it before trusting it
  elsewhere.
- The default split holds out the *last* 25% of pokes, and this take ramps up
  (34, 55, 78, 91, 87 N). Validation is therefore partly an **extrapolation** test
  — deliberately, but it does make the numbers look worse than a random poke split
  would. Say which you used.
