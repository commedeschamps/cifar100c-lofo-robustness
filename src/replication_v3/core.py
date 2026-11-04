"""Frozen protocol and durable artifact utilities. No automatic network operations."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (52, 67, 999, 555, 1666)
CONDITIONS = ("clean_baseline", "lofo_noise", "lofo_blur", "lofo_weather", "lofo_digital", "all_four")
REPRESENTATIVES = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
EXCLUDED = {"lofo_noise": 0, "lofo_blur": 1, "lofo_weather": 2, "lofo_digital": 3}
FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
MAIN_SPLIT = Path("splits/replication_v3_split.json")
INITIAL_ROOT = Path("artifacts/replication_v3/initializations")
GOLDEN_PATH = Path("artifacts/replication_v3/golden_corruptions.json")
LOCK_PATH = Path("configs/replication_v3/protocol.lock.json")
ROOT = Path("results/replication_v3")


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(obj):
    return hashlib.sha256(obj if isinstance(obj, bytes) else canonical(obj)).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(1024*1024), b""):
            h.update(c)
    return h.hexdigest()


def stable_seed(*parts):
    """Independent 32-bit deterministic stream, portable across workers and OSes."""
    raw = canonical(list(parts))
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8, person=b"lofo-v3").digest()[:4], "little")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path, payload):
    """Publish a complete file, never expose a partially written destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def atomic_json(path, obj):
    atomic_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode()+b"\n")


def atomic_torch(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp,"wb") as f:
            torch.save(obj,f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def atomic_copy(src, dst):
    src,dst = Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=dst.name+".", suffix=".tmp",dir=dst.parent)
    try:
        with open(src,"rb") as fi, os.fdopen(fd,"wb") as fo:
            shutil.copyfileobj(fi,fo,1024*1024)
            fo.flush();os.fsync(fo.fileno())
        os.replace(tmp,dst)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_csv(path, rows):
    import io
    if not rows:
        return
    dest=io.StringIO(newline="")
    w=csv.DictWriter(dest,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows)
    atomic_bytes(path,dest.getvalue().encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_info():
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True,stderr=subprocess.DEVNULL).strip()
        return commit,bool(dirty)
    except (OSError,subprocess.CalledProcessError):
        return None,True


def source_hash():
    """All scientific sources, excluding docs / mutable data and outputs."""
    paths=[]
    for name in ("src/replication_v3", "src/corruptions.py", "src/model.py", "src/lofo.py", "src/benchmark/data.py", "src/utils.py"):
        p=Path(name)
        if p.is_file():paths.append(p)
        elif p.is_dir():paths.extend(p.glob("*.py"))
    if not paths or not Path("src/model.py").exists():
        raise FileNotFoundError("Run from the existing research_cv repository root, after extracting this patch.")
    h=hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).replace("\\","/").encode());h.update(p.read_bytes().replace(b"\r\n",b"\n"))
    return h.hexdigest()


def proposal():
    """Explicitly descriptive, shared-operator dose-matched 4-slot experiment."""
    return {
        "protocol_version": "replication_v3.0.0",
        "study_status": "post_exploratory_controlled_replication",
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": 0.0,
        "num_classes": 100,
        "architecture": "cifar_resnet18_scratch",
        "train_per_class": 450,
        "val_per_class": 50,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
        "augmentation": {
            "design": "four_slots_independent_exposure",
            "sampling": "P(clean_slot)=0.5; P(each four representative slot)=0.125",
            "representatives": list(REPRESENTATIVES),
            "lofo_excluded_slot": "identity_no_op",
            "spatial": "zero-pad4_randomcrop32_random_horizontal_flip_0.5",
            "severity": "uniform_integer_1_to_5",
            "one_operator_per_presentation": True,
            "limitation": "Each retained operator has equal unconditional exposure (1/8) in LOFO and all-four; LOFO has 5/8 clean presentations, all-four 1/2, so total corrupted exposure is not matched. Causal interpretation remains limited."
        },
        "precision": "FP32; AMP disabled",
        "deterministic_algorithms": True,
        "selection": "first_strict_maximum_clean_val_accuracy",
        "split": str(MAIN_SPLIT).replace("\\","/"),
        "initializations": str(INITIAL_ROOT).replace("\\","/"),
        "golden": str(GOLDEN_PATH).replace("\\","/"),
        "train_uses_test": False,
    }


def prepare_split(data_root="data"):
    """Create exact-pixel disjoint fit/validation split, 450/50 per class.

    All repeated pixel contents remain in fit; validation consists of image-content
    singletons. Does not inspect test or benchmark data to choose the split.
    """
    from torchvision.datasets import CIFAR100
    raw=CIFAR100(str(data_root),train=True,download=False)
    digest_counts={}
    digests=[]
    for row in raw.data:
        h=hashlib.sha256(row.tobytes()).hexdigest()
        digests.append(h);digest_counts[h]=digest_counts.get(h,0)+1
    rng=random.Random(424242)
    validation=[]
    for cls in range(100):
        eligible=[i for i,y in enumerate(raw.targets) if y==cls and digest_counts[digests[i]]==1]
        if len(eligible)<50: raise ValueError(f"Class {cls} has only {len(eligible)} content-unique validation candidates")
        validation.extend(sorted(rng.sample(eligible,50)))
    val=set(validation)
    train=[i for i in range(len(raw)) if i not in val]
    if len(train)!=45000 or len(validation)!=5000:raise AssertionError("Incorrect split size")
    fit_hashes={digests[i] for i in train}
    if any(digests[i] in fit_hashes for i in validation):raise AssertionError("Exact-pixel leakage")
    result={"version":"replication_v3_split_1","selection_seed":424242,
            "method":"50 class-stratified unique-content validation samples; all duplicates in fit",
            "train_indices":train,"val_indices":sorted(validation),
            "train_count":45000,"val_count":5000,"pixel_sha256_overlap":0}
    if MAIN_SPLIT.exists() and sha_bytes(read_json(MAIN_SPLIT))!=sha_bytes(result):
        raise FileExistsError("Existing v3 split differs; never overwrite it silently")
    atomic_json(MAIN_SPLIT,result)
    return sha_file(MAIN_SPLIT)


def prepare_initials():
    import torch
    from src.model import build_model
    from src.utils import seed_everything
    INITIAL_ROOT.mkdir(parents=True,exist_ok=True)
    entries={}
    for seed in SEEDS:
        p=INITIAL_ROOT/f"seed_{seed}.pt"
        seed_everything(seed,deterministic=True)
        model=build_model().cpu()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if not p.exists():atomic_torch(p,state)
        else:
            current=torch.load(p,map_location="cpu",weights_only=True)
            if any(k not in current or not torch.equal(v,current[k]) for k,v in state.items()):
                raise ValueError(f"Initial tensors for seed {seed} differ from existing file")
        entries[str(seed)]={"path":str(p).replace("\\","/"),"sha256":sha_file(p)}
    existing=INITIAL_ROOT/"manifest.json"
    if existing.exists() and read_json(existing)!=entries:
        raise FileExistsError("Existing initialization manifest differs")
    atomic_json(existing,entries)
    return entries


def _golden_image():
    import numpy as np
    from PIL import Image
    y,x=np.indices((32,32))
    return Image.fromarray(np.stack([(13*x+7*y)%256,(5*x+19*y)%256,(x*x+y*y)%256],axis=-1).astype(np.uint8),"RGB")


def golden():
    import numpy as np
    from src.corruptions import apply_corruption
    values={}
    for op in REPRESENTATIVES:
        for severity in range(1,6):
            image=apply_corruption(_golden_image(),op,severity,np.random.RandomState(71422+severity))
            values[f"{op}:{severity}"]=sha_bytes(np.asarray(image).tobytes())
    return {"version":"v3_golden_1","operators":values}


def generate_golden():
    g=golden()
    if GOLDEN_PATH.exists() and read_json(GOLDEN_PATH)!=g:
        raise ValueError("Golden outputs differ from the committed reference")
    atomic_json(GOLDEN_PATH,g)
    return sha_file(GOLDEN_PATH)


def validate_golden():
    if not GOLDEN_PATH.is_file():raise FileNotFoundError("Golden file missing; generate on Mac and copy to Windows")
    g=golden()
    if g!=read_json(GOLDEN_PATH):
        raise RuntimeError("Platform corruption pixel outputs differ from reference. STOP: do not train")
    return True


def make_freeze(allow_dirty=False):
    """Freeze one source+data+golden+initial contract, then copy the lock unchanged."""
    if not MAIN_SPLIT.is_file() or not GOLDEN_PATH.is_file():
        raise FileNotFoundError("prepare-split, prepare-inits and make-golden first")
    initial_file=INITIAL_ROOT/"manifest.json"
    if not initial_file.is_file():raise FileNotFoundError("Initializations missing")
    commit,dirty=git_info()
    if (dirty or commit is None) and not allow_dirty:
        raise RuntimeError("Commit the scientific source locally before freezing (no push required)")
    for seed,entry in read_json(initial_file).items():
        if sha_file(entry["path"])!=entry["sha256"]:raise ValueError("Initialization checksum mismatch")
    validate_golden()
    base={"design":proposal(),"split_file_sha256":sha_file(MAIN_SPLIT),
          "initial_manifest_sha256":sha_file(initial_file),"golden_file_sha256":sha_file(GOLDEN_PATH),
          "source_sha256":source_hash(),"git_commit":commit,"git_dirty_at_freeze":dirty,
          "cifar_train_sha256":sha_file(Path("data")/"cifar-100-python"/"train"),
          "cifar_meta_sha256":sha_file(Path("data")/"cifar-100-python"/"meta")}
    frozen={**base,"protocol_sha256":sha_bytes(base)}
    if LOCK_PATH.exists() and read_json(LOCK_PATH)!=frozen:raise FileExistsError("Frozen protocol already exists and differs")
    atomic_json(LOCK_PATH,frozen)
    return frozen


def validate_freeze(require_git=True):
    if not LOCK_PATH.exists():raise FileNotFoundError("Protocol is not frozen. Run 'freeze' after author approval")
    lock=read_json(LOCK_PATH)
    h=lock.pop("protocol_sha256")
    if sha_bytes(lock)!=h:raise ValueError("Protocol lock has been modified")
    lock["protocol_sha256"]=h
    if lock["design"]!=proposal():raise ValueError("Protocol design differs from local implementation")
    for path,field in ((MAIN_SPLIT,"split_file_sha256"), (INITIAL_ROOT/"manifest.json","initial_manifest_sha256"),(GOLDEN_PATH,"golden_file_sha256")):
        if sha_file(path)!=lock[field]:raise ValueError(f"Frozen file differs: {path}")
    for seed,entry in read_json(INITIAL_ROOT/"manifest.json").items():
        if sha_file(entry["path"])!=entry["sha256"]:raise ValueError(f"Wrong initialization for seed {seed}")
    if source_hash()!=lock["source_sha256"]:raise ValueError("Frozen source identity mismatch")
    for f,field in ((Path("data")/"cifar-100-python"/"train","cifar_train_sha256"),
                    (Path("data")/"cifar-100-python"/"meta","cifar_meta_sha256")):
        if sha_file(f)!=lock[field]:raise ValueError(f"Official training dataset differs: {f}")
    if require_git:
        commit,dirty=git_info()
        if dirty:
            raise RuntimeError("Tracked source/worktree is dirty: commit reproducibility files locally first")
        # HEAD may change when protocol.lock.json itself is committed after freeze.
        # The checked source/content digests are authoritative, not this commit pointer.
    validate_golden()
    return lock


def machine_root(machine):
    if not machine or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in machine):
        raise ValueError("machine ID must be lowercase letters, digits, underscores and hyphens")
    return ROOT/machine


def run_dir(machine,seed,condition):
    if seed not in SEEDS or condition not in CONDITIONS:raise ValueError("Run not in registered grid")
    return machine_root(machine)/f"seed_{seed}"/condition


def safe_device(name):
    import torch
    if name not in ("cuda","mps"):raise ValueError("Only explicit cuda/mps production devices are supported")
    if name=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA not available; no CPU fallback")
    if name=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS not available; no CPU fallback")
    return torch.device(name)


def environment(device):
    import torch, torchvision, numpy as np
    from PIL import __version__ as pillow
    try:
        import cv2
        cv=cv2.__version__
    except ImportError: cv=None
    return {"platform":platform.platform(),"python":platform.python_version(),"torch":str(torch.__version__),
            "torchvision":torchvision.__version__,"numpy":np.__version__,"pillow":pillow,"opencv":cv,
            "device":str(device),"cuda_version":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else "Apple MPS",
            "cuda_tf32":False,"amp":False,"num_workers":2}


def assert_disk(path, min_gb=3):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(path).free
    if free<min_gb*1024**3:raise RuntimeError(f"Low disk space: {free/1024**3:.2f} GiB free")
    return free
