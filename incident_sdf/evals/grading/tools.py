"""Host-side tools for the grading agent: list_dir, read_file, submit_pick (brief §14.1).

The trial directory is the whole world: no network, no shell, no paths outside it. Picks are
recorded by the host in arrival order and are immutable (a second pick for the same question
is logged as ``duplicate`` and ignored). The model's narrative is stored separately and never
consulted for the picks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trials import Trial, resolve_pick

TOOL_SPECS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List file names in a directory under the work directory.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file under the work directory.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "submit_pick", "description": "Record your pick for one question.",
                                      "parameters": {"type": "object", "properties": {"qidx": {"type": "integer"}, "pick": {"type": "string"}},
                                                     "required": ["qidx", "pick"]}}},
]


class Sandbox:
    def __init__(self, work_dir: Path, trial: Trial):
        self.work = work_dir.resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "answers.csv").write_text(trial.csv_text, encoding="utf-8")
        self.trial = trial
        self.picks: dict[int, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def _resolve(self, path: str) -> Path:
        p = (self.work / path.lstrip("/")).resolve() if not path.startswith(str(self.work)) else Path(path).resolve()
        if p != self.work and self.work not in p.parents:
            raise PermissionError(f"path outside the work directory: {path}")
        return p

    def call(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "list_dir":
                p = self._resolve(args.get("path", "."))
                out = sorted(x.name for x in p.iterdir())
            elif name == "read_file":
                p = self._resolve(args["path"])
                out = p.read_text(encoding="utf-8")
            elif name == "submit_pick":
                qidx = int(args["qidx"])
                pick = str(args["pick"]).strip()
                truth = resolve_pick(self.trial, qidx, pick)
                if truth is None:
                    out = {"status": "invalid", "qidx": qidx, "pick": pick}
                elif qidx in self.picks:
                    out = {"status": "duplicate_ignored", "qidx": qidx, "pick": pick}
                else:
                    self.picks[qidx] = {"qidx": qidx, "pick": pick, "truth": truth, "order": len(self.picks)}
                    out = {"status": "recorded", "qidx": qidx, "pick": pick}
            else:
                out = {"status": "unknown_tool", "name": name}
        except (PermissionError, FileNotFoundError, KeyError, ValueError) as e:
            out = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        self.events.append({"tool": name, "args": args, "result_preview": str(out)[:200]})
        return out if isinstance(out, str) else json.dumps(out)
