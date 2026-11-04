"""Self-contained queue, artifact and scientific-sampling tests."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from src.replication_v3 import core
from src.replication_v3 import orchestrator
from src.replication_v3.__main__ import _classify, import_kit,export_kit


def test_design_is_auditable():
    design=core.proposal()
    assert tuple(design["seeds"]) == (52, 67, 999, 555, 1666)
    assert len(design["conditions"])==6
    assert design["epochs"]==100
    assert "5/8" in design["augmentation"]["limitation"]
    assert len(core.FAMILIES)==4
    assert sum(map(len,core.FAMILIES.values()))==15


def test_stateless_seed_non_python_hash():
    ref=core.stable_seed("branch",52,13,1409)
    assert ref==core.stable_seed("branch",52,13,1409)
    assert 0<=ref<2**32
    assert ref!=core.stable_seed("branch",67,13,1409)
    assert ref!=core.stable_seed("severity",52,13,1409)


def test_atomic_json_does_not_leave_partial_file(tmp_path):
    target=tmp_path/"some"/"state.json"
    core.atomic_json(target,{"epoch":32})
    core.atomic_json(target,{"epoch":33})
    assert core.read_json(target)=={"epoch":33}
    assert list(target.parent.glob("*.tmp"))==[]


def test_cross_platform_run_grid():
    assert core.run_dir("mac_m5",52,"clean_baseline")==Path("results/replication_v3/mac_m5/seed_52/clean_baseline")
    with pytest.raises(ValueError):core.run_dir("../escapes",52,"clean_baseline")
    with pytest.raises(ValueError):core.run_dir("mac_m5",42,"clean_baseline")


def test_sqlite_state_survives_restart(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with orchestrator.db_connect("mac_m5") as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]==30
        orchestrator.change(db,52,"clean_baseline","RUNNING",new_attempt=True)
    with orchestrator.db_connect("mac_m5") as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]==30
        assert db.execute("SELECT attempts FROM runs WHERE seed=52 AND condition='clean_baseline'").fetchone()[0]==1
        orchestrator.change(db,52,"clean_baseline","COMPLETED")
    assert sum(r["state"]=="COMPLETED" for r in orchestrator.get_rows("mac_m5"))==1


def test_pause_requested(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    orchestrator.pause("mac_m5")
    assert core.read_json(core.machine_root("mac_m5")/"pause.request")["requested"]


def test_fatal_errors_do_not_request_auto_hyperparam_change():
    assert _classify(RuntimeError("CUDA out of memory"))=="cuda_oom"
    assert _classify(ValueError("Protocol checksum mismatch"))=="protocol_mismatch"
    assert _classify(RuntimeError("device mps unavailable"))=="missing_accelerator"


def test_freeze_integrity_detects_tampering(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    core.LOCK_PATH.parent.mkdir(parents=True)
    original={"design":core.proposal(),"source_sha256":"fake"}
    core.atomic_json(core.LOCK_PATH,{**original,"protocol_sha256":core.sha_bytes(original)})
    with patch.object(core,"sha_file",return_value="fake"),patch.object(core,"source_hash",return_value="fake"),patch.object(core,"validate_golden",return_value=True):
        # Check that an edited field invalidates the self-hash before any file checks.
        file=core.read_json(core.LOCK_PATH)
        file["design"]["epochs"]=25
        core.atomic_json(core.LOCK_PATH,file)
        with pytest.raises(ValueError,match="modified"):core.validate_freeze(require_git=False)


def test_import_rejects_mismatched_bundle(tmp_path,monkeypatch):
    import zipfile
    monkeypatch.chdir(tmp_path)
    bad=tmp_path/"bad.zip"
    with zipfile.ZipFile(bad,"w") as z:
        z.writestr("replication_v3_export_manifest.json",json.dumps({"files":{"../bad_file":"deadbeef"}}))
        z.writestr("../bad_file",b"x")
    with pytest.raises(ValueError):import_kit(bad)
    assert not (tmp_path.parent/"bad_file").exists()


def test_train_integration_requires_project_sources():
    """Integration is deliberately checked in the real repository, never mocked as a success."""
    if not Path("src/model.py").exists() or not Path("src/corruptions.py").exists():
        pytest.skip("Install patch into existing research_cv repository for source-integration checks")
    from src.replication_v3.train import AugmentedCIFAR
    import numpy as np
    from types import SimpleNamespace
    data=np.zeros((6,32,32,3),dtype=np.uint8)
    data[:,8:24,8:24,:]=64
    raw=SimpleNamespace(data=data,targets=[0,1,2,3,4,5])
    a=AugmentedCIFAR(raw,[0,1,2,3,4,5],52,1,"lofo_noise",core.proposal()["mean"],core.proposal()["std"])
    x=[a[i] for i in range(len(a))]
    y=[a[i] for i in range(len(a))]
    import torch
    for p,q in zip(x,y):
        assert torch.equal(p[0],q[0]);assert p[1:]==q[1:]


def test_memmap_dataset_worker_pickle_is_small():
    import pickle
    from src.replication_v3.data import ArrayStore
    ds=ArrayStore("data/large_images.npy","data/labels.npy")
    ds._data=b"x"*(1024*1024)  # Simulate accidental loaded buffer in parent.
    encoded=pickle.dumps(ds)
    assert len(encoded)<1000
    loaded=pickle.loads(encoded)
    assert loaded._data is None
    assert loaded.images_path=="data/large_images.npy"
