"""hier_train 재개 체크포인트 — last.pt 저장·복원·설정 불일치 거부."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from greenguide_classifier import hier_train


def _tiny() -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    # 옵티마이저 상태(모멘트)가 생기도록 한 스텝
    loss = model(torch.randn(8, 4)).sum()
    loss.backward()
    opt.step()
    return model, opt


def test_save_then_load_restores_state(tmp_path: Path) -> None:
    model, opt = _tiny()
    hist = [{"epoch": 1, "val_coarse_acc": 0.5}, {"epoch": 2, "val_coarse_acc": 0.6}]
    path = tmp_path / "last.pt"
    hier_train.save_last_checkpoint(
        path, epoch=2, model=model, optimizer=opt, history=hist,
        best_score=0.61, best_epoch=2, patience_counter=0,
    )
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()

    model2, opt2 = _tiny()
    with torch.no_grad():
        model2.weight.add_(1.0)
    state = hier_train.load_last_checkpoint(path, model2, opt2, "cpu")
    assert state == {"epoch": 2, "history": hist, "best_score": 0.61, "best_epoch": 2, "patience_counter": 0}
    assert torch.equal(model2.weight, model.weight)
    for a, b in zip(opt2.state_dict()["state"].values(), opt.state_dict()["state"].values(), strict=True):
        assert torch.equal(a["exp_avg"], b["exp_avg"])


def test_missing_file_returns_none(tmp_path: Path) -> None:
    model, opt = _tiny()
    assert hier_train.load_last_checkpoint(tmp_path / "none.pt", model, opt, "cpu") is None


def test_config_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    model, opt = _tiny()
    path = tmp_path / "last.pt"
    hier_train.save_last_checkpoint(
        path, epoch=1, model=model, optimizer=opt, history=[],
        best_score=-1.0, best_epoch=-1, patience_counter=0,
    )
    # 다른 백본 설정으로 읽으면 이어 받지 않는다
    monkeypatch.setattr(hier_train, "BACKBONE", "resnet50-other")
    assert hier_train.load_last_checkpoint(path, model, opt, "cpu") is None
