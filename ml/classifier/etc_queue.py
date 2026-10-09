"""etc 큐 자동 처리 — open-set recognition 2단계 파이프라인.

etc('기타/분류 불가') 피드백이 임계(ETC_QUEUE_TRIGGER)에 도달하면:

  Stage 1) 기존 클래스 재배정
    etc 이미지를 현재 모델로 다시 보고, 임베딩이 기존 클래스 prototype 에
    충분히 가깝고(거리) + softmax 도 확신하면 그 클래스로 재배정.
    (사용자가 애매해서 '기타' 누른 케이스 회수. 신경망은 OOD 에 과신하므로
     softmax 단독이 아니라 임베딩 거리도 함께 본다.)

  Stage 2) 신규 클래스 후보
    어느 기존 클래스에도 안 붙는 것들끼리 HDBSCAN 클러스터링.
    뭉치면 숨김 pseudo-class(slug=etc_auto_*, active=false) 로 등록,
    흩어진 noise 는 etc 로 유지.

이후 retrain.py 가 재배정/pseudo 라벨로 학습 → 진단 게이트(④) 통과해야 활성화.
pseudo-class 는 active=false 라 사용자(/labels)에겐 안 보임. 운영자가
etc_clusters 리뷰 테이블을 보고 이름·배출법을 넣어 active=true 로 승격한다.

⚠️ 픽셀 군집에서 올바른 한국어 카테고리 이름은 자동 생성 불가 → 이름 짓기만
   사람이 한다 (그 외 전 과정 자동).
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
from greenguide_common import imaging, settings
from greenguide_common.logging import fail_open, get_logger
from PIL import Image
from postgrest.types import CountMethod
from sklearn.cluster import HDBSCAN
from supabase import Client

from greenguide_classifier import config
from greenguide_classifier.dataset import load_manifest
from greenguide_classifier.infer import softmax
from greenguide_classifier.model import build_model
from greenguide_classifier.train import model_kind, pick_device

log = get_logger(__name__)

# ── 임계값 (모두 보수적; 운영하며 보정) ──────────────────
ETC_QUEUE_TRIGGER = 30          # etc 피드백이 이만큼 쌓이면 처리 시작
REASSIGN_SOFTMAX = 0.85         # Stage1 재배정 softmax 확신 임계
REASSIGN_MAX_COSDIST = 0.40     # Stage1 prototype 까지 cosine 거리 상한
CLUSTER_MIN_SIZE = 5            # HDBSCAN min_cluster_size (신규 후보 최소 크기)
PROTO_SAMPLE_PER_CLASS = 40     # prototype 계산용 클래스당 샘플 수

_PSEUDO_PREFIX = "etc_auto_"
_IMG_SIZE = 224
# ImageNet 정규화 — greenguide_classifier/dataset.py 의 학습 transform 과 동일해야 함
_MEAN = imaging.MEAN_ARRAY
_STD = imaging.STD_ARRAY


def is_pseudo_slug(slug: str) -> bool:
    return slug.startswith(_PSEUDO_PREFIX)


def _supabase() -> Client:
    from greenguide_common.supabase import get_client  # noqa: PLC0415

    return get_client()


def count_etc_queue(client: Client | None = None) -> int:
    client = client or _supabase()
    res = (
        client.table(settings.SUPABASE_TABLE_USER_UPLOADS)
        .select("id", count=CountMethod.exact)
        .eq("feedback_label", "etc")
        .execute()
    )
    return res.count or 0


# ── 전처리 / 임베딩 ──────────────────────────────────────
def _to_tensor(img: Image.Image) -> np.ndarray:
    im = img.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # (3,H,W)


def _load_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """(logit_model, feature_extractor) — best.pt 로드.
    feature_extractor: ResNet18 backbone 의 avgpool 출력(512d).
    """
    ckpt_path = config.arch_subdir(config.CHECKPOINTS_DIR, "cnn") / "best.pt"
    model = build_model(model_kind("cnn")).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # backbone 의 fc 직전까지 = 512d feature
    feat = nn.Sequential(*list(model.backbone.children())[:-1]).to(device).eval()
    return model, feat


@torch.no_grad()
def _embed_batch(
    tensors: list[np.ndarray], model: nn.Module, feat: nn.Module, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """(embeddings (N,512) L2정규화, softmax (N,C))."""
    x = torch.from_numpy(np.stack(tensors)).to(device)
    emb = feat(x).flatten(1).cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    logits = model(x).cpu().numpy()
    probs = softmax(logits, axis=1)
    return emb, probs


def _download_image(url: str) -> Image.Image | None:
    with fail_open(log, f"이미지 다운로드 {url[:60]}…"):
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    return None


# ── prototype (기존 클래스 중심) ─────────────────────────
def _class_prototypes(
    feat: nn.Module, device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """학습 데이터에서 클래스별 평균 임베딩(L2정규화). (protos (K,512), labels)."""
    items = load_manifest()
    by_label: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)

    labels = sorted(by_label)
    protos = []
    rng = np.random.default_rng(config.SPLIT_SEED)
    for label in labels:
        group = by_label[label]
        idx = rng.permutation(len(group))[:PROTO_SAMPLE_PER_CLASS]
        tensors = []
        for i in idx:
            try:
                img = Image.open(group[i]["source_path"])
                tensors.append(_to_tensor(img))
            except Exception:  # noqa: BLE001 — fail-open: 손상 이미지는 prototype 에서 제외
                continue
        if not tensors:
            protos.append(np.zeros(512, dtype=np.float32))
            continue
        with torch.no_grad():
            x = torch.from_numpy(np.stack(tensors)).to(device)
            e = feat(x).flatten(1).cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        c = e.mean(axis=0)
        protos.append(c / (np.linalg.norm(c) + 1e-8))
    return np.stack(protos), labels


# ── 2단계 처리 ───────────────────────────────────────────
def process_etc_queue(apply: bool = False) -> dict[str, Any]:
    """etc 큐를 2단계로 처리. apply=False 면 dry-run(변경 없음, 분석만)."""
    client = _supabase()
    rows = (
        client.table(settings.SUPABASE_TABLE_USER_UPLOADS)
        .select("id,image_url,feedback_label")
        .eq("feedback_label", "etc")
        .execute()
    ).data or []
    n = len(rows)
    log.info(f"etc 피드백 {n}건 (트리거 {ETC_QUEUE_TRIGGER}) — apply={apply}")
    if n == 0:
        return {"count": 0, "reassigned": {}, "clusters": [], "noise": 0}

    device = pick_device()
    model, feat = _load_models(device)

    # 이미지 로드
    imgs, kept_rows = [], []
    for r in rows:
        img = _download_image(r["image_url"])
        if img is not None:
            imgs.append(_to_tensor(img))
            kept_rows.append(r)
    if not imgs:
        log.warning("유효 이미지 0 — 중단")
        return {"count": n, "reassigned": {}, "clusters": [], "noise": 0}

    emb, probs = _embed_batch(imgs, model, feat, device)
    protos, proto_labels = _class_prototypes(feat, device)

    # Stage 1) 기존 클래스 재배정
    reassigned: dict[str, str] = {}          # upload_id → class
    remaining_idx: list[int] = []
    for i, r in enumerate(kept_rows):
        cls_i = int(probs[i].argmax())
        conf = float(probs[i, cls_i])
        cls_label = config.CLASS_LABELS[cls_i] if cls_i < len(config.CLASS_LABELS) else None
        # prototype 거리 (cosine distance = 1 - cos sim)
        cosdist = float(1.0 - emb[i] @ protos[proto_labels.index(cls_label)]) \
            if cls_label in proto_labels else 1.0
        if (cls_label and not is_pseudo_slug(cls_label)
                and conf >= REASSIGN_SOFTMAX and cosdist <= REASSIGN_MAX_COSDIST):
            reassigned[r["id"]] = cls_label
        else:
            remaining_idx.append(i)

    # Stage 2) 신규 후보 클러스터링 (남은 것들)
    clusters: list[dict[str, Any]] = []
    noise = 0
    if len(remaining_idx) >= CLUSTER_MIN_SIZE:
        sub = emb[remaining_idx]
        labels_hd = HDBSCAN(
            min_cluster_size=CLUSTER_MIN_SIZE, metric="euclidean",
        ).fit_predict(sub)
        by_cluster: dict[int, list[int]] = defaultdict(list)
        for j, cl in enumerate(labels_hd):
            if cl == -1:
                noise += 1
            else:
                by_cluster[int(cl)].append(remaining_idx[j])
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for k, members in sorted(by_cluster.items()):
            slug = f"{_PSEUDO_PREFIX}{ts}_{k}"
            ids = [kept_rows[m]["id"] for m in members]
            clusters.append({"slug": slug, "size": len(members), "upload_ids": ids})
    else:
        noise = len(remaining_idx)

    summary = {
        "count": n,
        "reassigned": reassigned,
        "reassigned_dist": dict(Counter(reassigned.values())),
        "clusters": clusters,
        "noise": noise,
    }
    _print_summary(summary)

    if apply:
        _apply(client, summary)
    else:
        log.info("[dry-run] 변경 없음 — 실제 적용하려면 apply=True")
    return summary


def _apply(client: Client, summary: dict[str, Any]) -> None:
    # 1) 재배정 — feedback_label 을 기존 클래스로
    for upload_id, cls in summary["reassigned"].items():
        client.table(settings.SUPABASE_TABLE_USER_UPLOADS).update(
            {"feedback_label": cls},
        ).eq("id", upload_id).execute()
    log.info(f"{len(summary['reassigned'])}건 기존 클래스 재배정")

    # 2) 신규 후보 — 숨김 pseudo-class 등록 + feedback_label 갱신 + 리뷰행
    for k, cl in enumerate(summary["clusters"]):
        slug = cl["slug"]
        client.table("waste_classes").upsert({
            "slug": slug,
            "sort_order": 900 + k,
            "display_name": f"미분류 유형 {k + 1} (검토 대기)",
            "summary": "자동 발견된 미분류 군집 — 운영자 검토/명명 대기",
            "bin": None,
            "how_to": [],
            "caution": [],
            "color_hex": "#9E9E9E",
            "icon_name": "help_outline",
            "trained_in_model": False,
            "active": False,   # 사용자에겐 숨김 — 이름 붙인 뒤 active=true 로 승격
        }, on_conflict="slug").execute()
        for upload_id in cl["upload_ids"]:
            client.table(settings.SUPABASE_TABLE_USER_UPLOADS).update(
                {"feedback_label": slug},
            ).eq("id", upload_id).execute()
        with fail_open(log, "etc_clusters 기록 (migration 005 필요?)"):
            client.table("etc_clusters").insert({
                "slug": slug,
                "size": cl["size"],
                "sample_upload_ids": cl["upload_ids"][:10],
                "status": "pending_review",
            }).execute()
    log.info(f"신규 pseudo-class {len(summary['clusters'])}개 등록 (active=false)")


def _print_summary(s: dict[str, Any]) -> None:
    print(f"  Stage1 재배정: {len(s['reassigned'])}건 {s['reassigned_dist']}")
    print(f"  Stage2 신규 후보 군집: {len(s['clusters'])}개 "
          f"{[(c['slug'], c['size']) for c in s['clusters']]}")
    print(f"  noise(etc 유지): {s['noise']}건")


def maybe_process() -> dict[str, Any] | None:
    """retrain 초입에서 호출 — 임계 도달 시에만 실제 처리(apply=True)."""
    client = _supabase()
    n = count_etc_queue(client)
    if n < ETC_QUEUE_TRIGGER:
        log.info(f"etc {n}건 < 임계 {ETC_QUEUE_TRIGGER} — skip")
        return None
    log.info(f"etc {n}건 ≥ 임계 {ETC_QUEUE_TRIGGER} — 자동 처리 시작")
    return process_etc_queue(apply=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="etc_queue")
    ap.add_argument("--apply", action="store_true", help="실제 적용 (기본: dry-run)")
    args = ap.parse_args()
    process_etc_queue(apply=args.apply)
