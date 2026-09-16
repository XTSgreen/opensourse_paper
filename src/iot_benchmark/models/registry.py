"""Machine-readable registry for the three formal model stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    stage: int
    purpose: str
    aliases: Tuple[str, ...]


FORMAL_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="UOT-IOT",
        stage=1,
        purpose="状态转移结构恢复与输运解释",
        aliases=(
            "UOT-IOT",
            "Hard-IOT",
            "pure-IOT",
            "M0-IOT-balanced-OT",
            "M1-IOT-UOT",
            "M2-IOT-no-shared-single-domain",
            "M3-IOT-without-division-feature",
            "IOT Method D",
            "Frozen IOT transport-EMT",
            "IOT flow_R",
        ),
    ),
    ModelSpec(
        name="GH-IOT",
        stage=2,
        purpose="背景异质性、有限监督和条件状态组成预测",
        aliases=(
            "HH-IOT",
            "HH-IOT-LP",
            "IOT-HRT",
            "Hybrid-IOT",
            "Gated-IOT-DM",
            "Gated-IOT-global-alpha",
        ),
    ),
    ModelSpec(
        name="PERSIST-IOT",
        stage=3,
        purpose="多时间点谱系持续和未来状态预测",
        aliases=("PERSIST-IOT", "PERSIST-IOT v2"),
    ),
)


FORMAL_MODEL_NAMES = tuple(spec.name for spec in FORMAL_MODELS)


def alias_to_formal_model() -> Dict[str, str]:
    """Return a deterministic alias-to-formal-name lookup."""

    mapping: Dict[str, str] = {}
    for spec in FORMAL_MODELS:
        for alias in spec.aliases:
            if alias in mapping and mapping[alias] != spec.name:
                raise ValueError(f"历史别名重复归属：{alias}")
            mapping[alias] = spec.name
    return mapping


def formal_model_for(alias: str) -> str:
    """Resolve one historical name to a formal model."""

    try:
        return alias_to_formal_model()[str(alias)]
    except KeyError as exc:
        raise KeyError(f"未登记的模型或历史别名：{alias}") from exc


def all_aliases() -> Tuple[str, ...]:
    """Return aliases in stable stage and declaration order."""

    return tuple(alias for spec in FORMAL_MODELS for alias in spec.aliases)


def registry_summary() -> dict[str, object]:
    """Return a JSON-serialisable registry summary."""

    return {
        "formal_model_count": len(FORMAL_MODELS),
        "formal_models": [
            {
                "name": spec.name,
                "stage": spec.stage,
                "purpose": spec.purpose,
                "aliases": list(spec.aliases),
            }
            for spec in FORMAL_MODELS
        ],
        "alias_count": len(all_aliases()),
    }
