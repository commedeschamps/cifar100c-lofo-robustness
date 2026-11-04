"""Stateless augmentation and epoch-transactional ResNet-18 training.

Uses existing research code for the audited architecture and corruption operators.
No official test dataset is ever loaded in this module.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as TF

from src.model import build_model
from src.corruptions import apply_corruption
from src.utils import capture_rng,restore_rng,seed_everything,synchronize
from .core import (REPRESENTATIVES,EXCLUDED,CONDITIONS,SEEDS,MAIN_SPLIT,INITIAL_ROOT,
    read_json,run_dir,machine_root,safe_device,environment,sha_file,stable_seed,atomic_json,atomic_torch,
    atomic_copy,write_csv,validate_freeze,assert_disk,now,sha_bytes)


class AugmentedCIFAR(Dataset):
    """Index/epoch-keyed augmentation; deterministic independently of worker ordering."""
    def __init__(self, raw, indices, seed, epoch, condition, mean, std):
        self.raw=raw
        self.indices=tuple(indices)
        self.seed=int(seed)
        self.epoch=int(epoch)
        self.condition=condition
        self.mean=tuple(mean);self.std=tuple(std)

    def __len__(self):return len(self.indices)

    def __getitem__(self,i):
        epoch,i = i if isinstance(i,tuple) else (self.epoch,i)
        sample_id=self.indices[i]
        image=Image.fromarray(self.raw.data[sample_id],"RGB")
        label=int(self.raw.targets[sample_id])
        # Spatial decisions are shared by seed/epoch/image across conditions.
        rng=np.random.RandomState(stable_seed("spatial",self.seed,epoch,sample_id))
        image=ImageOps.expand(image,border=4,fill=0)
        top=int(rng.randint(0,9));left=int(rng.randint(0,9))
        image=image.crop((left,top,left+32,top+32))
        if rng.random_sample()<.5:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        # Draw once from five independent equal-probability paths.
        # [0..3] = four reps (each 1/8); [4] = clean (1/2).
        branch=np.random.RandomState(stable_seed("branch",self.seed,epoch,sample_id)).random_sample()
        slot=int(branch*8)
        op=REPRESENTATIVES[slot] if slot<4 else "clean"
        if self.condition=="clean_baseline" or (self.condition in EXCLUDED and slot==EXCLUDED[self.condition]):
            op="clean"
        severity=0
        if op!="clean":
            severity=1+int(np.random.RandomState(stable_seed("severity",self.seed,epoch,sample_id)).randint(5))
            operator_rng=np.random.RandomState(stable_seed("parameters",self.seed,epoch,sample_id,op))
            image=apply_corruption(image,op,severity,operator_rng)
        tensor=TF.to_tensor(image)
        tensor=TF.normalize(tensor, self.mean, self.std)
        return tensor,label,0 if op=="clean" else 1+REPRESENTATIVES.index(op),int(severity)


class CleanCIFAR(Dataset):
    def __init__(self,raw,indices,mean,std):
        self.raw=raw;self.indices=tuple(indices);self.mean=mean;self.std=std
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        index=self.indices[i]
        image=Image.fromarray(self.raw.data[index],"RGB")
        return TF.normalize(TF.to_tensor(image),self.mean,self.std),int(self.raw.targets[index])


class EpochSampler(Sampler):
    """Pass (epoch, index) to spawned persistent workers; order is epoch-keyed."""
    def __init__(self, size,seed,epoch):
        self.size=size;self.seed=seed;self.epoch=epoch
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return self.size
    def __iter__(self):
        generator=torch.Generator().manual_seed(stable_seed("shuffle",self.seed,self.epoch))
        for i in torch.randperm(self.size,generator=generator).tolist():
            yield (self.epoch,i)


def make_loaders(raw,split,design,seed,epoch,condition,device,workers=2,smoke_batches=None):
    train_idx=split["train_indices"]
    val_idx=split["val_indices"]
    if smoke_batches is not None:
        train_idx=train_idx[:max(128,smoke_batches*design["batch_size"])]
        val_idx=val_idx[:min(len(val_idx),256)]
    tr=AugmentedCIFAR(raw,train_idx,seed,epoch,condition,design["mean"],design["std"])
    va=CleanCIFAR(raw,val_idx,design["mean"],design["std"])
    sampler=EpochSampler(len(tr),seed,epoch)
    common={"batch_size":design["batch_size"],"num_workers":workers,"pin_memory":device.type=="cuda",
            "persistent_workers":workers>0,"drop_last":False}
    # The loader is constructed ONCE for the whole run; epoch IDs are carried
    # in sampler indices so persistent Windows spawn workers never use stale epoch state.
    g=torch.Generator().manual_seed(stable_seed("loader",seed))
    tr_dl=DataLoader(tr,sampler=sampler,generator=g,**common)
    val_dl=DataLoader(va,shuffle=False,**common)
    return tr_dl,val_dl,sampler


def single_epoch(model,loader,device,opt=None,max_batches=None):
    training=opt is not None
    model.train(training)
    criterion=nn.CrossEntropyLoss()
    n=0;loss_sum=0.;correct=0;counts=collections.Counter();updates=0
    context=torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_n,batch in enumerate(loader):
            if max_batches is not None and batch_n>=max_batches:break
            x,y=batch[:2]
            if training:
                for op_id,sev in zip(batch[2].tolist(),batch[3].tolist()):
                    name="clean" if op_id==0 else REPRESENTATIVES[op_id-1]
                    counts[name if name=="clean" else f"{name}:severity{sev}"]+=1
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            if training:opt.zero_grad(set_to_none=True)
            pred=model(x)
            if not torch.isfinite(pred).all():raise FloatingPointError("Non-finite logits")
            loss=criterion(pred,y)
            if not math.isfinite(float(loss.item())):raise FloatingPointError("Non-finite loss")
            if training:
                loss.backward()
                opt.step();updates+=1
            c=int(y.shape[0]);n+=c
            loss_sum+=float(loss.detach().item())*c
            correct+=int((pred.argmax(1)==y).sum().item())
    synchronize(device)
    if n==0:raise RuntimeError("Empty epoch")
    return {"loss":loss_sum/n,"accuracy_pct":correct*100.0/n,"correct":correct,"samples":n,
            "updates":updates,"exposure_counts":dict(counts)}


def selected(history):
    if not history:raise ValueError("Empty history")
    return max(history,key=lambda r:r["val_accuracy_pct"])


def load_safe(path):
    return torch.load(path,weights_only=True,map_location="cpu")


def verify_checkpoint(ckpt,manifest,expected_epochs=100):
    if ckpt.get("format")!="replication_v3_checkpoint_1":raise ValueError("Wrong checkpoint format")
    if ckpt.get("manifest")!=manifest:raise ValueError("Checkpoint manifest/identity differs")
    hist=ckpt["history"]
    epoch=ckpt["epoch"]
    if not 1<=epoch<=expected_epochs or epoch!=len(hist):raise ValueError("Wrong completed epoch")
    if [x["epoch"] for x in hist]!=list(range(1,epoch+1)):raise ValueError("Nonconsecutive training history")
    if any(not math.isfinite(x["val_accuracy_pct"]) or not math.isfinite(x["train_loss"]) for x in hist):
        raise ValueError("Non-finite history")
    best=selected(hist)
    if ckpt["best_epoch"]!=best["epoch"] or ckpt["best_val_accuracy_pct"]!=best["val_accuracy_pct"]:
        raise ValueError("Best checkpoint selection mismatch")
    if not ckpt.get("rng_state") or not ckpt.get("optimizer_state_dict") or not ckpt.get("scheduler_state_dict"):
        raise ValueError("Missing resume states")
    if not ckpt.get("model_state_dict") or not ckpt.get("exposure_history"):
        raise ValueError("Missing model/exposure audit")
    if len(ckpt["exposure_history"])!=epoch:raise ValueError("Incomplete exposure audit")
    allowed={"clean",*(f"{op}:severity{sev}" for op in REPRESENTATIVES for sev in range(1,6))}
    excluded=EXCLUDED.get(manifest["condition"])
    for row in ckpt["exposure_history"]:
        counts=row["counts"]
        if set(counts)-allowed or sum(counts.values())!=row["consumed"]:
            raise ValueError("Incomplete/invalid consumed corruption audit")
        if "smoke_batches" not in manifest and row["consumed"]!=45000:
            raise ValueError("Full epoch did not consume all 45,000 training images")
        if manifest["condition"]=="clean_baseline" and any(k!="clean" for k in counts):
            raise ValueError("Clean baseline received corruption")
        if excluded is not None and any(k.startswith(REPRESENTATIVES[excluded]+":") for k in counts):
            raise ValueError("Excluded LOFO operator appeared in training")
    if int(ckpt["scheduler_state_dict"].get("last_epoch",-1))!=epoch:
        raise ValueError("Scheduler does not match completed epoch")
    return True


def verify_training(machine,seed,condition,strict=True):
    out=run_dir(machine,seed,condition)
    manifest=read_json(out/"run_manifest.json")
    full=out/"checkpoints/last_checkpoint.pt"
    best_path=out/"checkpoints/best_model.pt"
    cp=load_safe(full)
    verify_checkpoint(cp,manifest)
    if strict and cp["epoch"]!=100:raise ValueError("Training not complete")
    best=load_safe(best_path)
    if best.get("epoch")!=cp["best_epoch"] or best.get("manifest")!=manifest:
        raise ValueError("Best weights do not match selected epoch")
    if best.get("val_accuracy_pct")!=cp["best_val_accuracy_pct"]:
        raise ValueError("Best validation score mismatch")
    if sha_file(INITIAL_ROOT/f"seed_{seed}.pt")!=manifest["initial_file_sha256"]:
        raise ValueError("Initial weights changed since training")
    if strict:
        completion=read_json(out/"completion_manifest.json")
        for name,hashvalue in completion["files"].items():
            if sha_file(out/name)!=hashvalue:raise ValueError(f"Completion artifact checksum mismatch: {name}")
        if completion["manifest_sha256"]!=sha_bytes(manifest):raise ValueError("Manifest checksum differs")
    return {"epoch":cp["epoch"],"best_epoch":cp["best_epoch"],"best_val_accuracy_pct":cp["best_val_accuracy_pct"]}


class StopRequested(Exception):pass

_STOP=False


def _signal_stop(_signum,_frame):
    global _STOP
    _STOP=True


def _manifest(lock,machine,seed,condition,device,workers):
    initial=read_json(INITIAL_ROOT/"manifest.json")[str(seed)]
    return {"protocol_sha256":lock["protocol_sha256"],"source_sha256":lock["source_sha256"],
        "git_commit":lock["git_commit"],"split_sha256":lock["split_file_sha256"],
        "initial_file_sha256":initial["sha256"],"machine":machine,"seed":seed,
        "condition":condition,"device":str(device),"environment":environment(device),
        "workers":int(workers)}


def _train_one(machine,seed,condition,device_name,workers=2,smoke_batches=None):
    """Full run is only possible against a frozen protocol. Smoke writes elsewhere."""
    global _STOP
    _STOP=False
    lock=validate_freeze(require_git=True)
    if condition not in CONDITIONS or seed not in SEEDS:raise ValueError("Run outside predefined grid")
    device=safe_device(device_name)
    out=run_dir(machine,seed,condition) if smoke_batches is None else Path("results/replication_v3_smoke")/machine/f"seed_{seed}"/condition
    out.mkdir(parents=True,exist_ok=True)
    assert_disk(out,3 if smoke_batches is None else 0.1)
    design=lock["design"]
    manifest=_manifest(lock,machine,seed,condition,device,workers)
    if smoke_batches is not None:manifest["smoke_batches"]=int(smoke_batches)
    mpath=out/"run_manifest.json"
    if mpath.exists():
        if read_json(mpath)!=manifest:raise ValueError("Existing run uses different frozen config/environment")
    elif any(p.name not in ("train.log", "error.json", ".run.lock") for p in out.iterdir()):
        raise ValueError("Run directory contains unrecognized data; refuse overwrite")
    else:atomic_json(mpath,manifest)
    if (out/"completion_manifest.json").exists():
        if smoke_batches is None:verify_training(machine,seed,condition)
        return "already_completed"
    seed_everything(seed,deterministic=True)
    from .data import ensure_cache
    raw=ensure_cache(verify_hash=True)
    split=read_json(MAIN_SPLIT)
    if len(split["train_indices"])!=45000 or len(split["val_indices"])!=5000:
        raise ValueError("Incorrect frozen split")
    init=INITIAL_ROOT/f"seed_{seed}.pt"
    if sha_file(init)!=manifest["initial_file_sha256"]:raise ValueError("Initialization digest mismatch")
    model=build_model()
    model.load_state_dict(load_safe(init),strict=True)
    model.to(device)
    opt=torch.optim.SGD(model.parameters(),lr=design["learning_rate"],momentum=design["momentum"],
                        weight_decay=design["weight_decay"],nesterov=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=design["epochs"],eta_min=design["scheduler_eta_min"])
    cdir=out/"checkpoints";cdir.mkdir(exist_ok=True)
    last=cdir/"last_checkpoint.pt";prev=cdir/"previous_checkpoint.pt";best=cdir/"best_model.pt"
    history=[];exposure=[];start_epoch=1;best_acc=-float("inf");best_epoch=0
    if last.exists() or prev.exists():
        chosen=None
        for candidate in (last,prev):
            if not candidate.exists():continue
            try:
                state=load_safe(candidate)
                verify_checkpoint(state,manifest)
                chosen=state
                if candidate==prev:
                    atomic_json(out/"recovery.json",{"at":now(),"recovered_from":"previous_checkpoint.pt",
                         "reason":"last missing or invalid","epoch":state["epoch"]})
                    atomic_copy(prev,last)
                break
            except Exception as e:
                atomic_json(out/f"{candidate.stem}_validation_error.json",{"error":str(e),"at":now()})
        if chosen is None:raise RuntimeError("No valid checkpoint; refusing to restart from zero")
        state=chosen
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng(state["rng_state"])
        history=state["history"];exposure=state["exposure_history"]
        best_acc=state["best_val_accuracy_pct"];best_epoch=state["best_epoch"]
        start_epoch=state["epoch"]+1
        if best_epoch==state["epoch"] and (not best.exists() or load_safe(best).get("epoch")!=best_epoch):
            atomic_torch(best,{"epoch":best_epoch,"val_accuracy_pct":best_acc,
                               "manifest":manifest,"model_state_dict":state["model_state_dict"]})
        if not best.exists() or load_safe(best).get("epoch")!=best_epoch:
            raise RuntimeError("Best checkpoint missing or stale; investigation required")
        # Last epoch checkpoint is authoritative after an interrupted artifact export.
        # Rebuild CSV/JSON here even if no more epochs remain to execute.
        write_csv(out/"history.csv",history)
        atomic_json(out/"corruption_counts.json",exposure)
    else:
        if best.exists():raise RuntimeError("Found best without last checkpoint; investigate")
    total_epochs=2 if smoke_batches is not None else design["epochs"]
    original_int=signal.getsignal(signal.SIGINT)
    original_term=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT,_signal_stop)
    signal.signal(signal.SIGTERM,_signal_stop)
    try:
        train_loader,val_loader,sampler=make_loaders(raw,split,design,seed,start_epoch,condition,device,
                                                     workers=workers,smoke_batches=smoke_batches)
        for epoch in range(start_epoch,total_epochs+1):
            if _STOP or (machine_root(machine)/"pause.request").exists():raise StopRequested("Pause requested")
            sampler.set_epoch(epoch)
            started=time.perf_counter()
            lr=opt.param_groups[0]["lr"]
            tr=single_epoch(model,train_loader,device,opt)
            va=single_epoch(model,val_loader,device)
            scheduler.step()
            row={"epoch":epoch,"learning_rate":lr,"train_loss":tr["loss"],
                 "train_accuracy_pct":tr["accuracy_pct"],"val_loss":va["loss"],
                 "val_accuracy_pct":va["accuracy_pct"],"updates":tr["updates"],
                 "seconds":time.perf_counter()-started}
            history.append(row)
            exposure.append({"epoch":epoch,"counts":tr["exposure_counts"],"consumed":tr["samples"]})
            is_best=va["accuracy_pct"]>best_acc
            if is_best:
                best_acc=va["accuracy_pct"];best_epoch=epoch
            state={"format":"replication_v3_checkpoint_1","manifest":manifest,"epoch":epoch,
                   "history":history,"exposure_history":exposure,"best_epoch":best_epoch,
                   "best_val_accuracy_pct":best_acc,"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":scheduler.state_dict(),
                   "rng_state":capture_rng()}
            verify_checkpoint(state,manifest)
            if last.exists():atomic_copy(last,prev)
            atomic_torch(last,state)
            if is_best:
                atomic_torch(best,{"epoch":epoch,"val_accuracy_pct":best_acc,"manifest":manifest,
                                   "model_state_dict":model.state_dict()})
            atomic_json(out/"progress.json",{"at":now(),"epoch":epoch,"total_epochs":total_epochs,
                 "seed":seed,"condition":condition,"train_accuracy_pct":tr["accuracy_pct"],
                 "val_accuracy_pct":va["accuracy_pct"],"last_epoch_seconds":row["seconds"],
                 "best_epoch":best_epoch,"device":device_name})
            write_csv(out/"history.csv",history)
            atomic_json(out/"corruption_counts.json",exposure)
            print(f"{machine} seed={seed} {condition} epoch={epoch}/{total_epochs} "
                  f"val={va['accuracy_pct']:.2f}% {row['seconds']:.1f}s",flush=True)
        if smoke_batches is not None:
            atomic_json(out/"smoke_complete.json",{"completed_epochs":len(history),"at":now()})
            return "smoke_completed"
        if len(history)!=100:raise ValueError("Incomplete run cannot be marked completed")
        proof={"protocol_sha256":lock["protocol_sha256"],"manifest_sha256":sha_bytes(manifest),
               "at":now(),"seed":seed,"condition":condition,"machine":machine,
               "best_epoch":best_epoch,"completed_epochs":100,
               "files":{str(p.relative_to(out)).replace("\\","/"):sha_file(p)
                        for p in (mpath,last,best,out/"history.csv",out/"corruption_counts.json")}}
        atomic_json(out/"completion_manifest.json",proof)
        verify_training(machine,seed,condition)
        return "completed"
    finally:
        signal.signal(signal.SIGINT,original_int)
        signal.signal(signal.SIGTERM,original_term)



def train_one(machine,seed,condition,device_name,workers=2,smoke_batches=None):
    """Prevent duplicate direct workers using a portable advisory lock."""
    from filelock import FileLock,Timeout
    out=run_dir(machine,seed,condition) if smoke_batches is None else Path("results/replication_v3_smoke")/machine/f"seed_{seed}"/condition
    out.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(out/".run.lock"),timeout=0):
            return _train_one(machine,seed,condition,device_name,workers,smoke_batches)
    except Timeout as e:
        raise RuntimeError(f"A worker already owns this run: {out}") from e
