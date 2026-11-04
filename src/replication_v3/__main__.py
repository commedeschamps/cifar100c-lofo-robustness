"""Single cross-platform entry point: python -m src.replication_v3 ..."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from .core import (INITIAL_ROOT,GOLDEN_PATH,LOCK_PATH,MAIN_SPLIT,SEEDS,CONDITIONS,now,
    prepare_split,prepare_initials,generate_golden,make_freeze,validate_freeze,
    machine_root,run_dir,read_json,sha_file,atomic_copy,atomic_json,safe_device)


def _cli():
    p=argparse.ArgumentParser(prog="python -m src.replication_v3",description="Controlled LOFO replication v3")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("plan")
    x=sub.add_parser("prepare-split");x.add_argument("--data-root",default="data")
    sub.add_parser("prepare-inits")
    sub.add_parser("prepare-cache")
    sub.add_parser("make-golden")
    x=sub.add_parser("freeze")
    x.add_argument("--confirm-design",action="store_true",help="Author confirms scientific design and its exposure limitation")
    x=sub.add_parser("export-kit");x.add_argument("--to",required=True)
    x=sub.add_parser("import-kit");x.add_argument("--from-file",required=True)
    for name in ("doctor","start","resume","status","pause","verify","backup","evaluate","benchmark"):
        x=sub.add_parser(name)
        x.add_argument("--machine",required=True)
        if name in ("doctor","start","resume","evaluate","benchmark"):
            x.add_argument("--device",choices=("mps","cuda"),required=True)
        if name in ("start","resume","benchmark"):
            x.add_argument("--workers",type=int,default=2)
        if name in ("start","resume"):
            x.add_argument("--max-retries",type=int,default=1)
            x.add_argument("--backup-to",default=None,help="Optional separate backup destination for each verified run")
            x.add_argument("--confirm-production",action="store_true",
                           help="Explicitly authorize all 30 full training runs on this machine")
        if name=="backup":x.add_argument("--to",required=True)
    x=sub.add_parser("aggregate")
    x.add_argument("--machines",nargs="+",required=True)
    x=sub.add_parser("worker")
    x.add_argument("--machine",required=True)
    x.add_argument("--device",choices=("mps","cuda"),required=True)
    x.add_argument("--seed",type=int,required=True)
    x.add_argument("--condition",choices=CONDITIONS,required=True)
    x.add_argument("--workers",type=int,default=2)
    x.add_argument("--smoke-batches",type=int,default=None)
    return p.parse_args()


def export_kit(target):
    """Portable frozen artifacts. Copy to Windows before the first benchmark."""
    validate_freeze(require_git=False)
    target=Path(target).resolve()
    paths=[LOCK_PATH,MAIN_SPLIT,GOLDEN_PATH,INITIAL_ROOT/"manifest.json"]
    paths.extend(INITIAL_ROOT/f"seed_{s}.pt" for s in SEEDS)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(f"Export already exists: {target}")
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED) as z:
        for path in paths:
            z.write(path,arcname=path.as_posix())
        z.writestr("replication_v3_export_manifest.json",
                    json.dumps({"created":now(),"files":{p.as_posix():sha_file(p) for p in paths}},indent=2))
    return {"bundle":str(target),"bytes":target.stat().st_size,"sha256":sha_file(target)}


def import_kit(source):
    """No zip-slip and no accidental overwrite; verify SHA before publishing."""
    source=Path(source)
    with zipfile.ZipFile(source) as z:
        manifest=json.loads(z.read("replication_v3_export_manifest.json"))
        approved=set(manifest["files"])
        expected={LOCK_PATH.as_posix(),MAIN_SPLIT.as_posix(),GOLDEN_PATH.as_posix(),
                  (INITIAL_ROOT/"manifest.json").as_posix()}
        expected.update((INITIAL_ROOT/f"seed_{s}.pt").as_posix() for s in SEEDS)
        if approved!=expected:raise ValueError("Bundle has unexpected scientific artifacts")
        from .core import sha_bytes,atomic_bytes
        # Read and verify all bytes BEFORE writing any artifact.
        data={}
        for name,digest in manifest["files"].items():
            raw=z.read(name)
            if sha_bytes(raw)!=digest:raise ValueError(f"Bundle corrupted: {name}")
            path=Path(name)
            if path.is_absolute() or ".." in path.parts:raise ValueError("Unsafe path")
            if path.exists() and sha_file(path)!=digest:
                raise FileExistsError(f"Different artifact exists; not overwriting: {path}")
            data[name]=raw
        for name,raw in data.items():
            p=Path(name)
            if not p.exists():atomic_bytes(p,raw)
    return {"imported":len(data),"bundle_sha256":sha_file(source)}


def _classify(exc):
    msg=str(exc).lower()
    if "out of memory" in msg or "cuda memory" in msg:return "cuda_oom"
    if "no such file" in msg or "not found" in msg or "dataset not found" in msg:return "data_missing"
    if "space left" in msg or "disk space" in msg:return "disk_full"
    if "unavailable" in msg and ("cuda" in msg or "mps" in msg):return "missing_accelerator"
    if any(t in msg for t in ("frozen","checkpoint","manifest","protocol","checksum","digest","dirty","source identity","golden","mismatch")):
        return "protocol_mismatch"
    if any(t in msg for t in ("nan","non-finite","nonfinite","invalid logits")):
        return "scientific_failure"
    return "unexpected_exit"


def main():
    args=_cli()
    cmd=args.command
    if cmd=="plan":
        print(json.dumps({"seeds":SEEDS,"conditions":CONDITIONS,"runs_per_machine":30,
                          "machines":2,"total_runs":60,
                          "scientific_limit":"5 distinct seeds repeated on MPS and CUDA, not n=10 independent seeds"},indent=2))
    elif cmd=="prepare-split":print("split SHA-256:",prepare_split(args.data_root))
    elif cmd=="prepare-inits":print(json.dumps(prepare_initials(),indent=2))
    elif cmd=="prepare-cache":
        from .data import ensure_cache
        print("Verified training-only memory-mapped cache:",ensure_cache(verify_hash=True).images_path)
    elif cmd=="make-golden":print("golden SHA-256:",generate_golden())
    elif cmd=="freeze":
        if not args.confirm_design:raise RuntimeError("Read/docs and explicitly acknowledge matched-operator/total-dose confound: use --confirm-design")
        print(json.dumps(make_freeze(),indent=2))
    elif cmd=="export-kit":print(json.dumps(export_kit(args.to),indent=2))
    elif cmd=="import-kit":print(json.dumps(import_kit(args.from_file),indent=2))
    elif cmd=="worker":
        from .train import train_one,StopRequested
        out=run_dir(args.machine,args.seed,args.condition)
        try:
            print(train_one(args.machine,args.seed,args.condition,args.device,args.workers,args.smoke_batches),flush=True)
        except StopRequested:
            print("Paused at epoch boundary",flush=True)
            raise SystemExit(75)
        except Exception as e:
            atomic_json(out/"error.json",{"at":now(),"category":_classify(e),"type":type(e).__name__,"message":str(e)})
            raise
    elif cmd=="aggregate":
        from .analyze import aggregate
        print(json.dumps(aggregate(args.machines),indent=2))
    elif cmd in ("doctor","start","resume","status","pause","verify","backup","evaluate","benchmark"):
        from .orchestrator import status,pause,preflight,start,verify_all,backup
        if cmd=="doctor":print(json.dumps(preflight(args.machine,args.device),indent=2))
        elif cmd in ("start","resume"):
            if not args.confirm_production:raise RuntimeError("Production disabled without --confirm-production")
            print("Queue result:",start(args.machine,args.device,args.workers,args.max_retries,args.backup_to))
        elif cmd=="status":status(args.machine)
        elif cmd=="pause":pause(args.machine)
        elif cmd=="verify":print(json.dumps(verify_all(args.machine),indent=2))
        elif cmd=="backup":print(json.dumps(backup(args.machine,args.to),indent=2))
        elif cmd=="evaluate":
            from .evaluate import evaluate
            evaluate(args.machine,args.device)
        elif cmd=="benchmark":
            preflight(args.machine,args.device)
            from .train import train_one
            import time
            st=time.perf_counter()
            result=train_one(args.machine,SEEDS[0],"lofo_noise",args.device,args.workers,smoke_batches=5)
            print(json.dumps({"result":result,"seconds":time.perf_counter()-st,
                "note":"Smoke 2 epochs with 5 train batches/epoch; not a scientific run."},indent=2))
    else:raise RuntimeError("Unknown command")


if __name__=="__main__":
    main()
