"""Predetermined redaction and flagging policy for trace text (brief §5.3).

Applied identically to every document before anything else looks at content. Two levels:
  * REPLACE   credentials, emails, full IPs, URL query secrets -> placeholder tokens
  * FLAG      operational-detail candidates (exploit payload patterns, live-target hosts)
              are flagged for the human review list, not removed automatically (PROPOSED:
              removing them blindly would change the corpus' character; the reviewer decides
              per flag and the decision is logged). A document with any replacement is
              ``transformed`` and labelled separately in the corpus manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

POLICY_VERSION = "redact-policy-v1"

REPLACE: list[tuple[str, re.Pattern[str], str]] = [
    ("credential", re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9\-]{10,})\b"), "[REDACTED_CREDENTIAL]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED_CREDENTIAL]"),
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED_CREDENTIAL]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    ("url_query_secret", re.compile(r"(?i)([?&](?:token|key|sig|signature|auth|apikey|api_key|access_token)=)[^&\s]+"), r"\1[REDACTED]"),
]

FLAG: list[tuple[str, re.Pattern[str]]] = [
    ("operational_detail", re.compile(r"(?i)(/etc/hosts|curl\s+-k|setsid\b|nc\s+-e|bash\s+-i\s*>&|/dev/tcp/|reverse shell|\{\{.*\}\}|jinja|hdf5|ssrf|xss|payload|exploit)")),
    ("live_target_host", re.compile(r"(?i)\b(?:[a-z0-9\-]+\.)+(?:windows\.net|amazonaws\.com|azure\.com|huggingface\.co|openai\.com|modal\.run)\b")),
]

# the publisher's own IP redaction ("20.9.x.x") must survive unchanged
_ALREADY_REDACTED_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.x\.x\b")


@dataclass
class RedactionResult:
    text: str
    transformed: bool
    replacements: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"transformed": self.transformed, "replacements": self.replacements,
                "flags": self.flags, "policy": POLICY_VERSION}


def redact(text: str) -> RedactionResult:
    out = text
    reps: dict[str, int] = {}
    for name, pat, repl in REPLACE:
        out, n = pat.subn(repl, out)
        if n:
            reps[name] = n
    flags: dict[str, int] = {}
    for name, pat in FLAG:
        n = len(pat.findall(out))
        if n:
            flags[name] = n
    log = [f"{POLICY_VERSION}: replaced {k}x{v}" for k, v in reps.items()]
    return RedactionResult(text=out, transformed=bool(reps), replacements=reps, flags=flags, log=log)
