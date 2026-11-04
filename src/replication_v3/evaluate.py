"""Resumable, official CIFAR-100-C per-condition inference after training.

Never used in model selection. Works on Windows (no fcntl/resource module).
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR100
from .core import (FAMILIES,SEEDS,CONDITIONS,machine_root,run_dir,read_json,now,
                   atomic_json,atomic_bytes,sha_file,validate_freeze,safe_device)
from .train import verify_training,load_safe


def _batches(model,raw,labels,device,batch_size):
    from src.benchmark.data import normalize_rgb
    output=np.empty(len(labels),dtype=np.uint8)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(labels),batch_size):
            j=min(i+batch_size,len(labels))
            x=normalize_rgb(raw[i:j],device)
            scores=model(x)
            if scores.shape!=(j-i,100) or not torch.isfinite(scores).all():
                raise RuntimeError("Invalid classification logits")
            output[i:j]=scores.argmax(1).detach().cpu().numpy().astype(np.uint8)
    return output


def _save_preds(path,values,metadata):
    import io
    f=io.BytesIO()
    np.savez_compressed(f,predictions=values,metadata=json.dumps(metadata,sort_keys=True))
    atomic_bytes(path,f.getvalue())


def _valid_pred(path,identity,targets):
    with np.load(path,allow_pickle=False) as cache:
        values=cache["predictions"]
        meta=json.loads(str(cache["metadata"]))
    if (meta["identity"]!=identity or values.shape!=(10000,) or values.dtype!=np.uint8
            or np.any(values>=100)):
        raise ValueError("Cached prediction identity/contents mismatched")
    count=int(np.count_nonzero(values==targets))
    if meta["n_correct"]!=count:raise ValueError("Cached correct count wrong")
    return count


def evaluate(machine,device_name,seed=None,condition=None):
    """Only call after the entire predetermined 30-run grid is verified."""
    lock=validate_freeze(require_git=True)
    from .orchestrator import verify_all
    ver=verify_all(machine)
    if ver["passed"]!=30:
        raise RuntimeError(f"Evaluation gated on all 30 complete trained conditions; {ver['passed']} complete")
    device=safe_device(device_name)
    from src.benchmark.data import CORRUPTIONS,inspect_benchmark,load_clean,severity_slice
    ds,clean_identity=load_clean("data")
    corruption_root,corruption_identity=inspect_benchmark("data",ds.targets)
    targets=np.asarray(ds.targets,dtype=np.int64)
    selected_seeds=SEEDS if seed is None else (seed,)
    selected_conditions=CONDITIONS if condition is None else (condition,)
    if any(s not in SEEDS for s in selected_seeds) or any(c not in CONDITIONS for c in selected_conditions):
        raise ValueError("Invalid grid selection")
    for s in selected_seeds:
        for c in selected_conditions:
            out=run_dir(machine,s,c)
            verify_training(machine,s,c)
            best=out/"checkpoints/best_model.pt"
            ckpt=load_safe(best)
            from src.model import build_model
            model=build_model()
            model.load_state_dict(ckpt["model_state_dict"],strict=True)
            model=model.to(device).eval()
            edir=out/"evaluation"
            edir.mkdir(parents=True,exist_ok=True)
            rows=[]
            dataset_id={"protocol_sha256":lock["protocol_sha256"],
                        "best_checkpoint_sha256":sha_file(best),
                        "test_sha256":clean_identity["test_sha256"],
                        "cifar_c_archive_md5":corruption_identity["archive_md5"]}
            # All 76 evaluation conditions are keyed by absolute corruption/severity identity.
            conditions=[("clean",0,None)]+[(name,sev,corruption_root/f"{name}.npy")
                                        for name in CORRUPTIONS for sev in range(1,6)]
            for name,severity,source in conditions:
                identity={**dataset_id,"corruption":name,"severity":severity,
                          "source_sha256":clean_identity["test_sha256"] if source is None
                              else corruption_identity["files"][f"{name}.npy"]["sha256"]}
                cache=edir/"predictions"/f"{name}__s{severity}.npz"
                if cache.exists():
                    count=_valid_pred(cache,identity,targets)
                    seconds=None
                else:
                    images=ds.data if source is None else np.load(source,mmap_mode="r",allow_pickle=False)[severity_slice(severity)]
                    started=time.perf_counter()
                    predictions=_batches(model,images,targets,device,128)
                    count=int(np.count_nonzero(predictions==targets))
                    seconds=time.perf_counter()-started
                    _save_preds(cache,predictions,{"identity":identity,"n_correct":count})
                    if _valid_pred(cache,identity,targets)!=count:raise RuntimeError("Prediction cache failed self-verification")
                rows.append({"machine":machine,"seed":s,"condition":c,"corruption":name,
                             "family":"clean" if name=="clean" else next(k for k,v in FAMILIES.items() if name in v),
                             "severity":severity,"n":10000,"n_correct":count,
                             "accuracy_pct":count/100.,"seconds":seconds})
                atomic_json(edir/"progress.json",{"at":now(),"completed":len(rows),"expected":76,
                                                    "condition":c,"seed":s})
            if len(rows)!=76:raise RuntimeError("Incomplete test evaluation")
            from .core import write_csv
            write_csv(edir/"per_condition.csv",rows)
            summary={"at":now(),"machine":machine,"seed":s,"condition":c,
                     "identity":dataset_id,"completed_conditions":76,
                     "clean_accuracy_pct":rows[0]["accuracy_pct"],
                     "mean_corrupted_accuracy_pct":sum(r["accuracy_pct"] for r in rows[1:])/75.,
                     "per_condition_sha256":sha_file(edir/"per_condition.csv")}
            atomic_json(edir/"summary.json",summary)
            print(f"EVALUATED {machine} {s} {c}: clean {summary['clean_accuracy_pct']:.2f} "
                  f"corrupted {summary['mean_corrupted_accuracy_pct']:.2f}",flush=True)
    return True
