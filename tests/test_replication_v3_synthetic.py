"""Real PyTorch interruption/restart drill, using tiny *synthetic* data.

No official CIFAR-100 test data and no production artifacts are involved.
"""
import sys
import types
from pathlib import Path
import numpy as np
import pytest
import torch


def test_2_epoch_crash_resume_matches_uninterrupted(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def build_model():
        return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,100))
    model_module=types.ModuleType("src.model")
    model_module.build_model=build_model
    corruption_module=types.ModuleType("src.corruptions")
    corruption_module.apply_corruption=lambda image,name,severity,rng:image
    utils_module=types.ModuleType("src.utils")
    def seed_everything(seed,deterministic=True):
        torch.manual_seed(seed);np.random.seed(seed)
        torch.set_num_threads(2)
    utils_module.seed_everything=seed_everything
    utils_module.capture_rng=lambda: {"torch":torch.get_rng_state()}
    utils_module.restore_rng=lambda x:torch.set_rng_state(x["torch"])
    utils_module.synchronize=lambda device:None
    for name,mod in (("src.model",model_module),("src.corruptions",corruption_module),("src.utils",utils_module)):
        monkeypatch.setitem(sys.modules,name,mod)
    from src.replication_v3 import core
    import importlib
    # This import happens after in-memory scientific-source stubs are installed.
    train=importlib.import_module("src.replication_v3.train")
    # In the full repository train.py may have been imported by another test.
    # Patch its captured scientific dependencies explicitly in either case.
    monkeypatch.setattr(train,"build_model",build_model)
    monkeypatch.setattr(train,"apply_corruption",lambda image,name,severity,rng:image)
    monkeypatch.setattr(train,"seed_everything",seed_everything)
    monkeypatch.setattr(train,"capture_rng",utils_module.capture_rng)
    monkeypatch.setattr(train,"restore_rng",utils_module.restore_rng)
    monkeypatch.setattr(train,"synchronize",utils_module.synchronize)
    initial=core.INITIAL_ROOT/"seed_52.pt"
    initial.parent.mkdir(parents=True)
    torch.manual_seed(52)
    core.atomic_torch(initial,build_model().state_dict())
    core.atomic_json(core.INITIAL_ROOT/"manifest.json",{"52":{"path":initial.as_posix(),"sha256":core.sha_file(initial)}})
    core.MAIN_SPLIT.parent.mkdir(parents=True)
    core.atomic_json(core.MAIN_SPLIT,{"train_indices":[i%400 for i in range(45000)],
                                       "val_indices":[400+i%100 for i in range(5000)]})
    torch.manual_seed(1)
    rng=np.random.RandomState(144)
    synthetic=types.SimpleNamespace(data=rng.randint(0,255,size=(500,32,32,3),dtype=np.uint8),
                                    targets=[i%100 for i in range(500)])
    class FakeCIFAR:
        def __init__(self,*args,**kwargs):self.data=synthetic.data;self.targets=synthetic.targets
    lock={"protocol_sha256":"synthetic_locked","source_sha256":"synthetic_source",
          "split_file_sha256":core.sha_file(core.MAIN_SPLIT),"git_commit":"test",
          "design":core.proposal()}
    monkeypatch.setattr(train,"validate_freeze",lambda require_git:lock)
    monkeypatch.setattr(train,"safe_device",lambda _:torch.device("cpu"))
    monkeypatch.setattr(train,"assert_disk",lambda *a,**kw:1024**4)
    from src.replication_v3 import data as v3_data
    monkeypatch.setattr(v3_data,"ensure_cache",lambda verify_hash=True:synthetic)
    monkeypatch.setattr(train,"environment",lambda d:{"device":"test_cpu","torch":str(torch.__version__)})
    calls={"n":0}
    real_single_epoch=train.single_epoch
    pause_path=core.machine_root("mac_m5")/"pause.request"
    def interrupt_after_validation(*args,**kwargs):
        result=real_single_epoch(*args,**kwargs)
        calls["n"]+=1
        if calls["n"]==2:
            core.atomic_json(pause_path,{"requested":"synthetic test"})
        return result
    monkeypatch.setattr(train,"single_epoch",interrupt_after_validation)
    with pytest.raises(train.StopRequested):
        train.train_one("mac_m5",52,"lofo_noise","mps",workers=0,smoke_batches=1)
    out=Path("results/replication_v3_smoke/mac_m5/seed_52/lofo_noise")
    first=train.load_safe(out/"checkpoints/last_checkpoint.pt")
    assert first["epoch"]==1
    pause_path.unlink()
    monkeypatch.setattr(train,"single_epoch",real_single_epoch)
    assert train.train_one("mac_m5",52,"lofo_noise","mps",workers=0,smoke_batches=1)=="smoke_completed"
    resumed=train.load_safe(out/"checkpoints/last_checkpoint.pt")
    assert resumed["epoch"]==2
    assert train.train_one("win_test",52,"lofo_noise","cuda",workers=0,smoke_batches=1)=="smoke_completed"
    fresh=train.load_safe(Path("results/replication_v3_smoke/win_test/seed_52/lofo_noise/checkpoints/last_checkpoint.pt"))
    assert resumed["history"]==fresh["history"] or all(
        all(a[k]==b[k] for k in ("epoch","train_loss","val_loss","val_accuracy_pct","updates"))
        for a,b in zip(resumed["history"],fresh["history"]))
    for k,v in resumed["model_state_dict"].items():
        assert torch.equal(v,fresh["model_state_dict"][k]),k
    assert resumed["exposure_history"]==fresh["exposure_history"]
