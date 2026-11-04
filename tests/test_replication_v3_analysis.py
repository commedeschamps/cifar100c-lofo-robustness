"""Statistical outputs must use 5 seed pairs per platform, not 10 pseudo-independent seeds."""
import csv
import sys
import types
from pathlib import Path
from src.replication_v3 import core


def test_analysis_counts_and_matched_operator_exposure(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.replication_v3 import analyze
    fake=types.ModuleType("src.replication_v3.train")
    fake.verify_training=lambda *args,**kw:{"verified":True}
    monkeypatch.setitem(sys.modules,"src.replication_v3.train",fake)
    conditions=list(core.CONDITIONS)
    names=[(c,sev) for fam in core.FAMILIES.values() for c in fam for sev in range(1,6)]
    assert len(names)==75
    for machine in ("mac_m5","win_rtx3050ti"):
        for seed in core.SEEDS:
            for condition in conditions:
                run=core.run_dir(machine,seed,condition)
                entries=[]
                for name,sev in [("clean",0),*names]:
                    val=7200 if name=="clean" else 4500
                    if condition!="clean_baseline" and name!="clean":
                        val+=150
                    if condition.startswith("lofo_") and name in core.FAMILIES[condition.split("_")[1]]:
                        val+=150
                    entries.append({"corruption":name,"severity":sev,"n_correct":val})
                core.write_csv(run/"evaluation/per_condition.csv",entries)
                counts={}
                if condition!="clean_baseline":
                    for j,op in enumerate(core.REPRESENTATIVES):
                        if core.EXCLUDED.get(condition)==j:continue
                        counts.update({f"{op}:severity{s}":100 for s in range(1,6)})
                core.atomic_json(run/"corruption_counts.json",[{"counts":counts,"epoch":1}])
                core.write_csv(run/"history.csv",[{"epoch":i+1,"train_loss":1.0-i*.001,
                              "val_accuracy_pct":75.0+i*.01} for i in range(100)])
    r=analyze.aggregate(["mac_m5","win_rtx3050ti"])
    assert r["runs"]==60
    assert r["paired_contrasts"]==40
    manifest=core.read_json(Path(r["output"])/"analysis_manifest.json")
    assert manifest["independent_training_seeds_per_condition"]==5
    assert manifest["observations_per_condition"]==10
    with (Path(r["output"])/"platform_paired_summary.csv").open() as f:
        d=list(csv.DictReader(f))
    assert len(d)==2*4*2
    assert set(x["n"] for x in d)=={"5"}
    assert all("ci95_low_t_approx" in x for x in d)
    with (Path(r["output"])/"matched_seed_platform_differences.csv").open() as f:
        assert len(list(csv.DictReader(f)))==30
