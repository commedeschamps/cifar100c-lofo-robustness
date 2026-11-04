# Windows RTX 3050 Ti (4 GB VRAM) - setup and overnight use

## Prerequisites

- Windows laptop connected to AC; 16 GB RAM; enough free **local SSD** space (recommend at least 20 GB beyond datasets).
- Python 3.11+ with a matching **CUDA**-enabled PyTorch/torchvision installation.
- NVIDIA drivers and CUDA-compatible PyTorch wheel. A CUDA toolkit installed globally is not necessary for standard PyTorch wheels.
- Same Git commit as the Mac's frozen source. Configure consistent LF line endings; source hashes normalize CRLF, but Python/operator behavior must still match.

From PowerShell, in the existing cloned repository:

```powershell
cd C:\path\to\research_cv
py -3.11 -m venv .venv-v3
.\.venv-v3\Scripts\Activate.ps1
python -m pip install -U pip
# Install matching CUDA PyTorch/torchvision following pytorch.org/get-started/locally
python -m pip install -r requirements-lofo.txt -r requirements-replication-v3.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m pytest -q tests/test_replication_v3*.py
```

Make sure CIFAR-100 is installed at `data/cifar-100-python/`. Extract the patch into the clone (or install with `install_patch.py`). Transfer the **frozen Mac kit** by USB/Drive, then:

```powershell
python -m src.replication_v3 import-kit --from-file "C:\path\to\lofo_v3_frozen_kit.zip"
```

The kit contains: protocol lock, same exact split, golden operator outputs and **same initial CPU model tensors** for all five seeds. Do not regenerate these on Windows. The code is not inside the kit and must be checked out from the same Git source.

## Mandatory CUDA preflight

```powershell
python -m src.replication_v3 prepare-cache
python -m src.replication_v3 doctor --machine win_rtx3050ti --device cuda
python -m src.replication_v3 benchmark --machine win_rtx3050ti --device cuda --workers 2
```

Check batch size 128 on the actual GPU. If it fails with CUDA OOM, do NOT silently reduce Windows batch size or enable AMP. Revisit and re-freeze the protocol for BOTH machines before starting the real 30-run grid.

Check that all 20 golden corruption hashes match the Mac. Differences in OpenCV/Pillow/JPEG implementations are a STOP condition and require an explicit decision, not silently accepting different training inputs.

## Night-time power settings

Windows Settings → System → Power & battery → Screen, sleep & hibernate timeouts: disable sleep **while plugged in**. The optional PowerShell commands below adjust AC timeouts; run with appropriate permissions and review your device policy:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Set lid-close action appropriately through Control Panel → Power Options; do not assume closed-lid training will work. Keep airflow clear and the charger connected.

## Train / monitor / pause

```powershell
python -m src.replication_v3 start --machine win_rtx3050ti --device cuda --workers 2 --confirm-production
```

Open `results\replication_v3\win_rtx3050ti\dashboard.html` in the browser for a local status snapshot. The queue refreshes its contents while running.

In a second PowerShell window:

```powershell
python -m src.replication_v3 status --machine win_rtx3050ti
python -m src.replication_v3 pause --machine win_rtx3050ti
```

After pausing or reopening Windows, rerun with `resume` replacing `start`. Completed runs are verified/skipped; interrupted runs resume at the latest completed epoch.

```powershell
python -m src.replication_v3 resume --machine win_rtx3050ti --device cuda --workers 2 --confirm-production
```

Once 30/30 complete:

```powershell
python -m src.replication_v3 verify --machine win_rtx3050ti
python -m src.replication_v3 evaluate --machine win_rtx3050ti --device cuda
```

Copy the **final verified Windows result directory** to Mac under the same `results/replication_v3/win_rtx3050ti/` relative path, using SHA-256-verifiable backups. Never copy Windows artifacts over Mac ones.
