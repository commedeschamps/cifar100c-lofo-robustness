"""Descriptive paired training-seed statistics, platform-blocked by construction."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import statistics as stats
from .core import SEEDS,CONDITIONS,FAMILIES,machine_root,run_dir,read_json,atomic_json,write_csv,now


def get_eval(machine,seed,condition):
    path=run_dir(machine,seed,condition)/"evaluation/per_condition.csv"
    if not path.exists():raise FileNotFoundError(f"Evaluation missing: {path}")
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=76:raise ValueError(f"Expected 76 conditions: {path}")
    keys={(r["corruption"],int(r["severity"])) for r in rows}
    if len(keys)!=76:raise ValueError(f"Duplicate conditions: {path}")
    return rows


def subset_score(rows,family=None,clean=False):
    if clean:
        selected=[r for r in rows if r["corruption"]=="clean"]
    elif family:
        selected=[r for r in rows if r["corruption"] in FAMILIES[family]]
    else:
        selected=[r for r in rows if r["corruption"]!="clean"]
    if len(selected)!=(1 if clean else len(FAMILIES[family])*5 if family else 75):
        raise ValueError("Incomplete family evaluation")
    return sum(int(r["n_correct"]) for r in selected)/(100.*len(selected))


def describe(vals):
    n=len(vals);mean=sum(vals)/n;sd=stats.stdev(vals) if n>1 else 0.
    # For n=5, 95% two-sided Student-t multiplier, df=4. Interpret cautiously.
    tcrit=2.7764451051977987 if n==5 else None
    margin=tcrit*sd/math.sqrt(n) if tcrit is not None else None
    return {"n":n,"mean":mean,"sd":sd,"min":min(vals),"max":max(vals),
            "ci95_low_t_approx":mean-margin if margin is not None else "n_not_5",
            "ci95_high_t_approx":mean+margin if margin is not None else "n_not_5"}




def verify_matched_operator_exposure(machine,seed):
    """Operator+severity exposure is *exactly* paired for all conditions retaining it."""
    from .core import REPRESENTATIVES,EXCLUDED
    totals={}
    for condition in CONDITIONS:
        exposure=read_json(run_dir(machine,seed,condition)/"corruption_counts.json")
        counter={}
        for epoch in exposure:
            for name,count in epoch["counts"].items():
                counter[name]=counter.get(name,0)+int(count)
        totals[condition]=counter
    reference=totals["all_four"]
    for condition in CONDITIONS:
        if condition=="clean_baseline":continue
        for j,op in enumerate(REPRESENTATIVES):
            if EXCLUDED.get(condition)==j:continue
            for sev in range(1,6):
                key=f"{op}:severity{sev}"
                if totals[condition].get(key,0)!=reference.get(key,0):
                    raise ValueError(f"Shared operator exposure mismatch: {machine}/{seed}/{condition}/{key}")
    return True


def aggregate(machines):
    from .train import verify_training
    if len(set(machines))!=len(machines):raise ValueError("Duplicate machine IDs")
    model_rows=[];diff_rows=[]
    for machine in machines:
        for seed in SEEDS:
            verify_matched_operator_exposure(machine,seed)
            data={}
            for cond in CONDITIONS:
                verify_training(machine,seed,cond)
                data[cond]=get_eval(machine,seed,cond)
                model_rows.append({"machine":machine,"seed":seed,"condition":cond,
                    "clean_accuracy_pct":subset_score(data[cond],clean=True),
                    "corrupted_accuracy_pct":subset_score(data[cond])})
            for family in FAMILIES:
                lofo="lofo_"+family
                baseline=subset_score(data["clean_baseline"],family=family)
                held=subset_score(data[lofo],family=family)
                allfour=subset_score(data["all_four"],family=family)
                denom=allfour-baseline
                diff_rows.append({"machine":machine,"seed":seed,"family":family,
                    "baseline_accuracy_pct":baseline,"heldout_lofo_accuracy_pct":held,
                    "all_four_accuracy_pct":allfour,"lofo_gain_pp":held-baseline,
                    "all_four_gain_pp":denom,"observed_transfer_ratio":(held-baseline)/denom if abs(denom)>1e-10 else "undefined"})
    grouped_models=[];grouped_diff=[];convergence=[]
    for machine in machines:
        for seed in SEEDS:
            for cond in CONDITIONS:
                path=run_dir(machine,seed,cond)/"history.csv"
                with path.open(newline="",encoding="utf-8") as f:
                    history=list(csv.DictReader(f))
                if len(history)!=100:raise ValueError(f"100-epoch convergence history missing: {path}")
                val=[float(r["val_accuracy_pct"]) for r in history]
                epochs=[float(r["train_loss"]) for r in history]
                best=max(range(100),key=lambda j:val[j])+1
                convergence.append({"machine":machine,"seed":seed,"condition":cond,"best_epoch":best,
                    "best_at_last_epoch":best==100,"last_validation_accuracy_pct":val[-1],
                    "val_accuracy_last10_delta_pp":val[-1]-val[-10],
                    "train_loss_last10_delta":epochs[-1]-epochs[-10]})
    for machine in machines:
        for cond in CONDITIONS:
            rs=[r for r in model_rows if r["machine"]==machine and r["condition"]==cond]
            for metric in ("clean_accuracy_pct","corrupted_accuracy_pct"):
                grouped_models.append({"machine":machine,"condition":cond,"metric":metric,
                                       **describe([r[metric] for r in rs])})
        for fam in FAMILIES:
            rs=[r for r in diff_rows if r["machine"]==machine and r["family"]==fam]
            for metric in ("lofo_gain_pp","all_four_gain_pp"):
                grouped_diff.append({"machine":machine,"family":fam,"metric":metric,
                                     **describe([r[metric] for r in rs])})
    root=Path("results/replication_v3/analysis")
    root.mkdir(parents=True,exist_ok=True)
    write_csv(root/"seed_level_model_results.csv",model_rows)
    write_csv(root/"paired_contrasts.csv",diff_rows)
    write_csv(root/"platform_model_summary.csv",grouped_models)
    write_csv(root/"platform_paired_summary.csv",grouped_diff)
    write_csv(root/"convergence_diagnostics.csv",convergence)
    # The same 5 seed IDs across platforms are not treated as n=10.
    cross=[]
    if len(machines)==2:
        for cond in CONDITIONS:
            for s in SEEDS:
                a,b=[next(r for r in model_rows if r["machine"]==m and r["seed"]==s and r["condition"]==cond)
                     for m in machines]
                cross.append({"seed":s,"condition":cond,"machine_a":machines[0],"machine_b":machines[1],
                              "clean_difference_pp":a["clean_accuracy_pct"]-b["clean_accuracy_pct"],
                              "corrupted_difference_pp":a["corrupted_accuracy_pct"]-b["corrupted_accuracy_pct"]})
        write_csv(root/"matched_seed_platform_differences.csv",cross)
    atomic_json(root/"analysis_manifest.json",{
        "at":now(),"machines":machines,"unique_seed_count":len(SEEDS),
        "observations_per_condition":len(SEEDS)*len(machines),
        "independent_training_seeds_per_condition":len(SEEDS),
        "interpretation":"Descriptive paired contrasts per platform, not ten independent-seed replicates.",
        "uncertainty":"Sample SD and approximate two-sided 95% t intervals (df=4) over five seeds within each platform; strong small-n assumptions; no significance claim.",
        "total_evaluated_runs":len(model_rows),"heldout_differences":len(diff_rows)})
    return {"runs":len(model_rows),"paired_contrasts":len(diff_rows),"output":str(root)}
