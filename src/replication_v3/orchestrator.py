"""Crash-recoverable, local-SQLite, one-worker-at-a-time orchestrator."""
from __future__ import annotations
import csv
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock,Timeout
from .core import (CONDITIONS,SEEDS,ROOT,machine_root,run_dir,now,atomic_json,
                   read_json,sha_file,atomic_copy,validate_freeze,assert_disk,safe_device)


@contextmanager
def db_connect(machine):
    root=machine_root(machine)
    root.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(root/"queue.sqlite3", timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=20000")
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        seed INTEGER NOT NULL, condition TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'PENDING', attempts INTEGER NOT NULL DEFAULT 0,
        started TEXT, finished TEXT, detail TEXT,
        PRIMARY KEY(seed, condition))""")
    for seed in SEEDS:
        for condition in CONDITIONS:
            conn.execute("INSERT OR IGNORE INTO runs(seed,condition) VALUES(?,?)",(seed,condition))
    conn.commit()
    try:yield conn
    finally:conn.close()


def change(conn,seed,cond,state,detail=None,new_attempt=False):
    conn.execute("""UPDATE runs SET state=?, detail=?, started=CASE WHEN ?='RUNNING' THEN ? ELSE started END,
        finished=CASE WHEN ? IN ('COMPLETED','FAILED','BLOCKED') THEN ? ELSE finished END,
        attempts=attempts+? WHERE seed=? AND condition=?""",
        (state,detail,state,now(),state,now(),int(new_attempt),seed,cond))
    conn.commit()


def get_rows(machine):
    with db_connect(machine) as conn:
        return [{"seed":seed,"condition":name,"state":state,"attempts":n,"detail":detail}
                for seed,name,state,n,detail in conn.execute("SELECT seed,condition,state,attempts,detail FROM runs ORDER BY seed,condition")]


def estimate(machine):
    """Approximate queue ETA from observed full/smoke-free completed epochs."""
    durations=[];completed_epochs=0
    for seed in SEEDS:
        for cond in CONDITIONS:
            path=run_dir(machine,seed,cond)/"history.csv"
            if not path.exists():continue
            try:
                with path.open(newline="",encoding="utf-8") as f:
                    rows=list(csv.DictReader(f))
                completed_epochs+=len(rows)
                durations.extend(float(r["seconds"]) for r in rows
                                 if 0<float(r["seconds"])<36000)
            except (OSError,ValueError,KeyError):continue
    # Run counts are fixed; partial runs contribute their already completed epochs.
    if not durations:return {"observed_epochs":completed_epochs,"eta_hours":None}
    from statistics import median
    seconds_per_epoch=median(durations[-min(len(durations),50):])
    return {"observed_epochs":completed_epochs,
            "seconds_per_epoch_recent_median":round(seconds_per_epoch,2),
            "eta_hours":round(max(0,3000-completed_epochs)*seconds_per_epoch/3600,1),
            "estimate_note":"Based on recent completed epochs; excludes evaluation, reboot and backup delays"}


def status(machine):
    rows=get_rows(machine)
    counts={}
    for row in rows:counts[row["state"]]=counts.get(row["state"],0)+1
    print(f"Machine: {machine} | runs: {len(rows)} | "+" | ".join(f"{key}={value}" for key,value in sorted(counts.items())))
    e=estimate(machine)
    if e["eta_hours"] is not None:
        print(f"Approx training ETA: {e['eta_hours']} hours (recent completed epochs only)")
    for row in rows:
        if row["state"] in ("RUNNING","FAILED","INTERRUPTED","BLOCKED","PAUSED"):
            d=run_dir(machine,row["seed"],row["condition"])
            info=""
            if (d/"progress.json").exists():
                p=read_json(d/"progress.json")
                info=f" epoch {p['epoch']}/{p['total_epochs']}, val {p['val_accuracy_pct']:.2f}%"
            print(f"{row['seed']:>5} {row['condition']:16} {row['state']:13}{info} {row['detail'] or ''}")
    return counts


def pause(machine):
    p=machine_root(machine)/"pause.request"
    atomic_json(p,{"requested":now()})
    print(f"Pause requested for {machine}. Current epoch will be allowed to finish.")


def preflight(machine,device):
    lock=validate_freeze(require_git=True)
    safe_device(device)
    free=assert_disk(machine_root(machine),min_gb=12)
    if not Path("data/cifar-100-python/train").exists():
        # torchvision will report missing data if this common-layout path does not exist;
        # other dataset layouts may be valid and will be loaded by the worker.
        from torchvision.datasets import CIFAR100
        CIFAR100("data",train=True,download=False)
    if not (Path("src")/"model.py").is_file():raise FileNotFoundError("Wrong repository root")
    return {"frozen":lock["protocol_sha256"],"device":device,"disk_gib_free":round(free/1024**3,2)}


def start(machine,device,workers=2, max_retries=1, backup_to=None):
    """Acquires a cross-process lock, owns local queue and launches isolated workers.

    Unknown failures retry once from a valid checkpoint. Explicit scientific/data
    failures stop the queue rather than changing any settings.
    """
    if max_retries not in (0,1,2):raise ValueError("0, 1, or 2 retries are supported")
    preflight(machine,device)
    root=machine_root(machine)
    lockfile=FileLock(str(root/".queue.lock"),timeout=0)
    try:lockfile.acquire(timeout=0)
    except Timeout as e:raise RuntimeError("Another queue process owns this machine") from e
    pause_file=root/"pause.request"
    if pause_file.exists():pause_file.unlink()
    state={"sigint":False}
    def request_pause(sig,frame):
        state["sigint"]=True
        if not pause_file.exists():pause(machine)
        print("Pause requested; worker should finish current epoch.",flush=True)
    old_int=signal.getsignal(signal.SIGINT)
    old_term=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT,request_pause)
    signal.signal(signal.SIGTERM,request_pause)
    try:
        with db_connect(machine) as conn:
            for seed in SEEDS:
                for cond in CONDITIONS:
                    if pause_file.exists():return "paused"
                    output=run_dir(machine,seed,cond)
                    # Durable completion marker is authoritative, not SQLite state.
                    if (output/"completion_manifest.json").exists():
                        from .train import verify_training
                        verify_training(machine,seed,cond)
                        change(conn,seed,cond,"COMPLETED","verified from existing outputs")
                        print(f"SKIP verified {seed} {cond}",flush=True)
                        continue
                    attempts=0
                    while attempts<=max_retries:
                        if pause_file.exists():return "paused"
                        assert_disk(root,3)
                        attempts+=1
                        change(conn,seed,cond,"RUNNING",f"attempt {attempts}",new_attempt=True)
                        output.mkdir(parents=True,exist_ok=True)
                        old_error=output/"error.json"
                        if old_error.exists():
                            atomic_copy(old_error,output/f"error_before_attempt_{attempts}.json")
                            old_error.unlink()
                        log=output/"train.log"
                        args=[sys.executable,"-m","src.replication_v3","worker","--machine",machine,
                              "--device",device,"--seed",str(seed),"--condition",cond,"--workers",str(workers)]
                        print(f"START {seed} {cond} attempt {attempts}: {now()}",flush=True)
                        from .dashboard import render
                        render(machine)
                        with log.open("a",encoding="utf-8",buffering=1) as f:
                            f.write(f"\n==== QUEUE START {now()} ATTEMPT {attempts} ====\n")
                            p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,
                                               cwd=Path.cwd(),stdin=subprocess.DEVNULL)
                            last_notice=0
                            while p.poll() is None:
                                time.sleep(5)
                                if time.monotonic()-last_notice>120:
                                    latest=output/"progress.json"
                                    if latest.is_file():
                                        try:
                                            progress=read_json(latest)
                                            print(f"RUN {seed} {cond} epoch {progress['epoch']}/{progress['total_epochs']}, "
                                                  f"last checkpoint {progress['at']}",flush=True)
                                        except (KeyError,ValueError,OSError):pass
                                    last_notice=time.monotonic()
                                    render(machine)
                            rc=p.returncode
                        if rc==0:
                            from .train import verify_training
                            try:
                                verify_training(machine,seed,cond)
                            except Exception as e:
                                change(conn,seed,cond,"FAILED",f"Completed worker failed verification: {e}")
                                raise
                            change(conn,seed,cond,"COMPLETED","verified")
                            print(f"DONE {seed} {cond}",flush=True)
                            render(machine)
                            if backup_to is not None:
                                try:
                                    result=backup_one(machine,seed,cond,backup_to)
                                    print(f"BACKUP {seed}/{cond}: {result}",flush=True)
                                except Exception as exc:
                                    print(f"BACKUP WARNING {seed}/{cond}: {exc}; continuing training",flush=True)
                            break
                        error=read_json(output/"error.json") if (output/"error.json").exists() else {}
                        category=error.get("category","unexpected_exit")
                        if rc==75 or pause_file.exists():
                            change(conn,seed,cond,"PAUSED","epoch-boundary pause")
                            return "paused"
                        if category in ("cuda_oom","data_missing","protocol_mismatch","scientific_failure",
                                        "missing_accelerator","disk_full"):
                            change(conn,seed,cond,"BLOCKED",f"fatal: {category}; {error.get('message','')}")
                            raise RuntimeError(f"Blocked: {seed}/{cond} {category}: {error.get('message','')}")
                        if attempts>max_retries:
                            change(conn,seed,cond,"FAILED",f"failed after {attempts} attempts: {error}")
                            raise RuntimeError(f"Run {seed}/{cond} failed; see {log}")
                        print(f"RETRY {seed}/{cond} from verified checkpoint after {category}",flush=True)
        return "completed"
    finally:
        signal.signal(signal.SIGINT,old_int)
        signal.signal(signal.SIGTERM,old_term)
        lockfile.release()


def verify_all(machine):
    from .train import verify_training
    success=[];fail=[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            try:success.append((seed,condition,verify_training(machine,seed,condition)))
            except Exception as e:fail.append({"seed":seed,"condition":condition,"error":str(e)})
    report={"machine":machine,"checked":now(),"passed":len(success),"expected":30,"failures":fail}
    atomic_json(machine_root(machine)/"verification_report.json",report)
    return report


def backup(machine,destination):
    """Copy only immutable verified completed outputs to an offline/Drive staging path."""
    from .train import verify_training
    dest=Path(destination).resolve()
    src_root=machine_root(machine).resolve()
    if dest==src_root or src_root in dest.parents:raise ValueError("Backup must be outside active result root")
    dest.mkdir(parents=True,exist_ok=True)
    report={"started":now(),"machine":machine,"copied":[],"skipped":[],"errors":[]}
    for seed in SEEDS:
        for cond in CONDITIONS:
            try:
                verify_training(machine,seed,cond)
            except Exception as e:
                report["skipped"].append({"seed":seed,"condition":cond,"reason":str(e)})
                continue
            rel=Path(machine)/f"seed_{seed}"/cond
            src=run_dir(machine,seed,cond)
            target=dest/rel
            if target.exists():
                existing=target/"completion_manifest.json"
                if existing.exists() and read_json(existing)==read_json(src/"completion_manifest.json"):
                    report["skipped"].append(str(rel));continue
                report["errors"].append({"path":str(rel),"error":"Backup destination already exists with different content"})
                continue
            staging=target.with_name(target.name+".partial")
            if staging.exists():shutil.rmtree(staging)
            try:
                staging.parent.mkdir(parents=True,exist_ok=True)
                staging.mkdir(parents=True)
                for path in src.rglob("*"):
                    if not path.is_file() or path.name==".run.lock":
                        continue
                    output=staging/path.relative_to(src)
                    output.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,output)
                    if sha_file(path)!=sha_file(output):raise IOError("Copy verification checksum failed")
                staging.rename(target)
                report["copied"].append(str(rel))
            except Exception as e:
                report["errors"].append({"path":str(rel),"error":str(e)})
                # Leave partial staging for inspection, never delete the source.
    report["finished"]=now()
    atomic_json(machine_root(machine)/"backup_report.json",report)
    return report


def backup_one(machine,seed,condition,destination):
    """Optional backup of one COMPLETED run; never affects queue success."""
    from .train import verify_training
    verify_training(machine,seed,condition)
    dest=Path(destination).resolve()
    source=run_dir(machine,seed,condition).resolve()
    if source==dest or dest in source.parents or source in dest.parents:
        raise ValueError("Backup destination must be separate from active run tree")
    target=dest/machine/f"seed_{seed}"/condition
    if target.exists():
        record=target/"completion_manifest.json"
        if record.exists() and read_json(record)==read_json(source/"completion_manifest.json"):
            return "already_backed_up"
        raise FileExistsError("Backup already exists but differs; refusing overwrite")
    stage=target.with_name(target.name+".partial")
    if stage.exists():
        raise FileExistsError(f"Prior backup staging exists; inspect before retry: {stage}")
    stage.mkdir(parents=True,exist_ok=False)
    for p in source.rglob("*"):
        if not p.is_file() or p.name==".run.lock":continue
        q=stage/p.relative_to(source)
        q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,q)
        if sha_file(p)!=sha_file(q):raise IOError(f"Backup checksum failed: {p}")
    target.parent.mkdir(parents=True,exist_ok=True)
    stage.rename(target)
    return "backed_up"
