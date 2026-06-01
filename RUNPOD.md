# RunPod Playbook — GENESIS Paper 5

Short recipe for pretraining + diffusion fine-tuning on RunPod spot GPUs.
All three variants (Small, Base, Large) train from scratch on RunPod — no
dependency on the Mac smoke test checkpoints.

---

## Recommended pod

**Small + Base:** RTX A5000 24 GB spot (~$0.22/hr).
**Large:** RTX 4090 or A6000 spot (~$0.34–0.50/hr).

- Template: **RunPod PyTorch 2.x**
- Disk: **50 GB** container + **50 GB** volume
- Connect: SSH over web terminal or `ssh root@<pod-ip>`

## 1. Upload code + data (once per pod)

This repo ships a pre-staged `runpod_upload/` folder at the Mac-side root that
contains **only** what the pod needs: the 6 training/eval scripts, the full
`src/model/` package, and the 8 `data/processed/` files required for training.
No Mac-only data-prep scripts, no DuckDB, no raw sources, no intermediate
curation artifacts. On-disk size: ~343 MB.

### Option A: runpodctl send/receive (recommended — no SSH keys needed)

**On your Mac:**
```bash
cd /Users/rakibhhridoy/AsGW/GroundWater/Paper5
tar czf /tmp/runpod_upload.tar.gz -C runpod_upload .
runpodctl send /tmp/runpod_upload.tar.gz
# Note the receive code (e.g. abc-def-ghi)
```

**On the pod (web terminal):**
```bash
# Install runpodctl if not present
curl -sSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -o /usr/local/bin/runpodctl && chmod +x /usr/local/bin/runpodctl

cd /workspace
runpodctl receive <code>
tar xzf runpod_upload.tar.gz
rm runpod_upload.tar.gz
```

Install `runpodctl` on Mac via `brew install runpod/runpodctl/runpodctl`.

### Option B: rsync over SSH (requires SSH key in RunPod settings)

Add your public key (`~/.ssh/id_ed25519.pub`) to RunPod → Settings → SSH
Public Keys, then restart the pod. Then from your Mac:

```bash
cd /Users/rakibhhridoy/AsGW/GroundWater/Paper5
rsync -avz --exclude='__pycache__' -e "ssh -p <PORT>" \
    runpod_upload/ root@<pod-ip>:/workspace/
```

Check the pod dashboard for the SSH port number.

---

On the pod, `/workspace/` will contain `src/`, `data/processed/`, and
the shell scripts at the root — identical layout to the Mac side, so
every `Path(__file__).parent.parent / "data" / "processed"` path in the
Python scripts resolves correctly without modification.

`checkpoints/`, `logs/`, and `results/` are **not** shipped — the training
scripts create them on first run. This guarantees a clean-slate fresh run
on the pod (no stray Mac smoketest weights).

## 2. Verify pod + run setup

```bash
ssh root@<pod-ip>
cd /workspace/Paper5
bash runpod_setup.sh          # installs deps, verifies GPU + all 8 data files
```

`runpod_setup.sh` is a strict gate: it fails fast if GPU VRAM < 24 GB, any of
the 8 required data files is missing or truncated, or `normalization_stats.json`
is not in canonical array format. Fix whatever it reports before launching
training — a bad data file surfaces here in ~30 seconds instead of 6 h into a
run.

## 3. Run the three variants (in order)

Each of the three scripts is **standalone** — Stage 1 → Stage 2 → Diffusion
→ Eval for one model size. There is no cross-size warm-starting: each size
trains from scratch against the same data. Run them sequentially so you can
inspect the previous size's results before spending on the next.

**Unattended mode** (auto-stops pod when done or at 12 h wall-clock):
```bash
mkdir -p logs
nohup timeout --signal=SIGTERM --kill-after=2m 12h \
    ./run_small.sh > logs/small.log 2>&1 &
tail -f logs/small.log
```

Each script has an EXIT trap that calls `runpodctl stop pod $RUNPOD_POD_ID`
on any exit (success, failure, or SIGTERM from `timeout`). Safe to launch
and disconnect — billing stops automatically.

**Attended mode** (manual stop):
```bash
mkdir -p logs

# 1. Small — fast gate (~6–8 h, ~$2 on A5000 spot)
nohup ./run_small.sh > logs/small.log 2>&1 &
tail -f logs/small.log
# Wait for completion. Inspect results/eval_small.json.

# 2. Base — main scaling-law data point (~30 h, ~$7 on A5000 spot)
nohup ./run_base.sh  > logs/base.log  2>&1 &
tail -f logs/base.log
# Wait for completion. Inspect results/scaling_summary.json — you should
# see Base MAEs lower than Small on most parameters. If not, stop here.

# 3. Large — only if Small → Base shows clear scaling
#    Switch to a 4090 / A6000 pod first (more VRAM, faster cores).
#    ~50 h, ~$17 on 4090 spot.
#    Use setsid+disown pattern — the 2026-05-03 / 2026-05-12 "stuck at
#    epoch 21" interruptions were SIGHUP on web-terminal disconnect.
#    Plain `nohup ... &` is not enough; setsid detaches from the session.
setsid nohup ./run_large.sh > logs/large.log 2>&1 < /dev/null &
disown
tail -f logs/large.log
```

Each script runs 4 steps for its variant:
1. Stage 1 pretrain (100 ep, full 2 M corpus)
2. Stage 2 fine-tune (50 ep, warm-start from that size's `stage1/best.pt`)
3. Diffusion fine-tune (300 ep, conditioned on the size's encoder)
4. Zero-shot transfer evaluation on Bangladesh held-out pairs

Results write to `results/eval_<size>.json` per variant and are also merged
into `results/scaling_summary.json` — so if you only run Small + Base (skip
Large), the summary file still contains the two-point scaling curve.

### If the pod is preempted

Spot pods can be preempted mid-run. The pretrain script writes `latest.pt`
every `--ckpt_every_steps` (default 500). The shell scripts' `pick_resume`
helper automatically picks up `latest.pt` if present, so just re-running
`./run_<size>.sh` after a preempt resumes from the interrupted step. If you
need to resume a single stage manually:

```bash
python src/04_pretrain_genesis.py \
    --size base --stage 1 --epochs 100 \
    --batch_size 512 --lr 1e-4 --mask_ratio 0.2 \
    --patience 20 --num_workers 4 --fp16 \
    --resume auto
```

`--resume auto` picks `checkpoints/base_stage1/latest.pt` if present, else
`best.pt`. Optimizer, scheduler, AMP scaler, RNG state, and step-in-epoch
are all restored — you do not lose progress within the interrupted epoch.
The diffusion script (`05_finetune_diffusion.py`) has the same `--resume`
flag with the same semantics.

## 4. Download results back

### Option A: runpodctl send/receive (no SSH keys needed)

**On the pod (web terminal):** restart the stopped pod first, then:
```bash
cd /workspace
tar czf results_small.tar.gz checkpoints/ results/ logs/
runpodctl send results_small.tar.gz
# Note the receive code
```

**On your Mac:**
```bash
cd /Users/rakibhhridoy/AsGW/GroundWater/Paper5
runpodctl receive <code>
tar xzf results_small.tar.gz
rm results_small.tar.gz
```

### Option B: download_from_pod.sh (requires SSH keys)

Trained artifacts go to the external SSD at `/Volumes/SSD Ex/GENESIS_runpod/`,
not the Mac internal disk. Use the helper from your Mac after each variant
finishes:

```bash
./download_from_pod.sh <pod-ip> small    # after run_small.sh finishes
./download_from_pod.sh <pod-ip> base     # after run_base.sh finishes
./download_from_pod.sh <pod-ip> large    # after run_large.sh finishes
# or pull everything at once:
./download_from_pod.sh <pod-ip> all
```

Layout on the SSD:

```
/Volumes/SSD Ex/GENESIS_runpod/
├── small/{checkpoints,logs}/
├── base/{checkpoints,logs}/
├── large/{checkpoints,logs}/
└── results/         # shared — eval_{size}.json + scaling_summary.json
```

The helper checks the SSD is mounted, separates per-variant checkpoints so
runs don't overwrite each other, and merges the `results/` directory across
variants. The Mac repo itself is never written to — storage stays clean.

---

Once the variant's artifacts are downloaded, **stop/terminate the pod** to
end billing. If unattended mode was used, the pod auto-stopped already —
just restart it briefly to download, then terminate.

## Cost summary

| Variant | GPU | Wall | Cost |
|---|---|---|---|
| Small (Stage 1 + 2 + Diffusion + Eval) | A5000 spot | ~6–8 h | ~$2 |
| Base  (Stage 1 + 2 + Diffusion + Eval) | A5000 spot | ~30 h  | ~$7 |
| Large (Stage 1 + 2 + Diffusion + Eval) | 4090 spot  | ~50 h  | ~$17 |
| Buffer (debugging, re-runs, preempts) | | | ~$5 |
| **Total (all three)** | | | **~$31** |

Inside the $100 outline budget with headroom. Running only Small + Base
(skipping Large) is ~$14 total and still gives a two-point scaling curve
for Figure 1d.

## Troubleshooting

- **CUDA OOM on Base batch 512:** drop to 256. Script handles this
  without re-spec; resume from `latest.pt` after lowering.
- **Spot preempted mid-epoch:** just re-run the same `./run_<size>.sh`.
  `pick_resume` picks up `latest.pt` automatically.
- **Dataset workers crashing on RunPod:** try `--num_workers 2`.
  `num_workers=4` is fine on Linux w/ enough RAM; not portable to Mac.
- **`torch.compile` slowdowns:** not enabled by default. Don't turn on
  without A/B testing — transformer of this size can be slower compiled.
