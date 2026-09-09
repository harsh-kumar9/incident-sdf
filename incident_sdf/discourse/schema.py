"""Episode bank / source packet schema for the discourse arm (brief §6.1, §20).

A packet is the model-visible bundle for one generation request: specific source
passages (with source id and evidence level), observed actions, stated rationales,
attributed interpretations, uncertainties, cross-source dependencies, and the allowed
factual claims. Ten retellings of one exchange are one episode, not ten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVIDENCE_LEVELS = ("observation", "agent_statement", "investigator_inference", "commentary")
FORMS = {
    "chronological_case_notes": "dated case notes that walk through what was observed, in order, "
                                "with the source of each observation named",
    "brief_technical_explanation": "a short technical explainer of how the observed mechanism worked, "
                                   "at the level of detail the sources support",
    "comparative_source_analysis": "a comparison of what the different sources report and where they "
                                   "agree, differ, or leave gaps",
    "retrospective_interpretation": "a retrospective that lays out the attributed interpretations and "
                                    "the uncertainties, keeping each interpretation with its author",
}


@dataclass
class Passage:
    passage_id: str
    source_id: str
    evidence_level: str
    text: str
    span_note: str = ""
    verbatim: list[str] = field(default_factory=list)   # exact substrings of the cited snapshot (faithfulness anchors)

    def __post_init__(self) -> None:
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(f"{self.passage_id}: bad evidence level {self.evidence_level!r}")


@dataclass
class Episode:
    episode_id: str
    incident_id: str
    canonical_event_cluster_id: str
    title: str
    source_ids: list[str]
    passages: list[Passage]
    observed_actions: list[str] = field(default_factory=list)
    expressed_rationales: list[str] = field(default_factory=list)
    attributed_interpretations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    cross_source_dependencies: list[str] = field(default_factory=list)
    allowed_factual_claims: list[str] = field(default_factory=list)
    split: str = "train"
    transformation_log: list[str] = field(default_factory=list)

    def passage_ids(self) -> set[str]:
        return {p.passage_id for p in self.passages}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bank(path: Path) -> list[Episode]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    eps = []
    for e in raw["episodes"]:
        eps.append(Episode(**{**e, "passages": [Passage(**p) for p in e["passages"]]}))
    ids = [e.episode_id for e in eps]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode ids in bank")
    for e in eps:
        for p in e.passages:
            if p.source_id not in e.source_ids:
                raise ValueError(f"{e.episode_id}: passage {p.passage_id} cites unlisted source {p.source_id}")
    return eps


def bank_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
