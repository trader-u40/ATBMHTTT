"""Execution oracle (JSDOM) and Pasini-style parser/DOM oracle."""

from __future__ import annotations

import json
import select
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from decode import XSS_HINT, injection_candidates

LAB_ROOT = Path(__file__).resolve().parents[1]
ORACLE_JS = LAB_ROOT / "oracle" / "jsdom_oracle.mjs"


def wrap_html(payload: str) -> str:
    return (
        "<html><head><title>lab</title></head><body>"
        + payload
        + "</body></html>"
    )


def parser_changed(payload: str, baseline: str = "abc") -> bool:
    """Pasini TH1/TH2 style: any DOM tree difference vs a benign string."""

    def signature(html: str) -> tuple:
        soup = BeautifulSoup(wrap_html(html), "html.parser")
        return tuple(
            (tag.name, tuple(sorted(tag.attrs.keys())))
            for tag in soup.find_all()
            if tag.name
        )

    return signature(payload) != signature(baseline)


@dataclass
class OracleResult:
    executed: bool
    parser_changed: bool
    candidate: str | None = None
    hooks: list[dict] = field(default_factory=list)
    timeout: bool = False
    error: str | None = None

    @property
    def browser_alive(self) -> bool:
        return self.executed

    @property
    def parser_alive(self) -> bool:
        return self.parser_changed


class JsdomOracle:
    def __init__(self, recycle_every: int = 800, timeout_s: float = 5.0) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._timeout_s = timeout_s
        self._calls = 0
        self._recycle_every = recycle_every
        self._err_path = LAB_ROOT / "data" / "jsdom_oracle.stderr.log"

    def start(self) -> None:
        if self._proc is not None:
            return
        self._err_path.parent.mkdir(parents=True, exist_ok=True)
        self._err = open(self._err_path, "a", encoding="utf-8")
        err = self._err
        self._proc = subprocess.Popen(
            ["node", str(ORACLE_JS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            cwd=str(LAB_ROOT),
            bufsize=1,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            if self._proc.stdout:
                self._proc.stdout.close()
            self._proc.wait(timeout=5)
            self._err.close()
        except Exception:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = None

    def __enter__(self) -> "JsdomOracle":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def check_snippet(
        self, snippet: str, request_id: int = 0, as_document: bool = False
    ) -> dict:
        snippet = snippet[:12000]
        self._calls += 1
        if self._calls % self._recycle_every == 0:
            self.close()
        last_err = None
        for _ in range(2):
            if self._proc is None:
                self.start()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            line = json.dumps(
                {
                    "id": request_id,
                    "payload": snippet,
                    "document": as_document,
                },
                ensure_ascii=False,
            )
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
                ready, _, _ = select.select([self._proc.stdout], [], [], self._timeout_s)
                if not ready:
                    self._proc.kill()
                    self.close()
                    return {"executed": False, "hooks": [], "timeout": True}
                raw = self._proc.stdout.readline()
            except BrokenPipeError as err:
                last_err = err
                self.close()
                continue
            if not raw:
                last_err = RuntimeError("jsdom oracle worker closed unexpectedly")
                self.close()
                continue
            return json.loads(raw)
        return {
            "executed": False,
            "hooks": [],
            "error": f"oracle_restart_failed:{last_err}",
        }

    def evaluate(self, payload: str) -> OracleResult:
        parser_hit = False
        last_error = None
        timeout = False
        candidates = injection_candidates(payload)
        if not any(XSS_HINT.search(c) for c in candidates):
            parser_hit = any(parser_changed(c) for c in candidates) or parser_changed(
                payload
            )
            return OracleResult(
                executed=False,
                parser_changed=parser_hit,
                candidate=None,
            )
        hinted = [c for c in candidates if XSS_HINT.search(c)][:4]
        for idx, candidate in enumerate(hinted):
            row = self.check_snippet(candidate, request_id=idx)
            if row.get("timeout"):
                timeout = True
            if row.get("error"):
                last_error = row.get("error")
            if row.get("executed"):
                return OracleResult(
                    executed=True,
                    parser_changed=True,
                    candidate=candidate,
                    hooks=row.get("hooks") or [],
                    timeout=timeout,
                    error=last_error,
                )
        parser_hit = any(parser_changed(c) for c in candidates[:3]) or parser_changed(
            payload
        )
        return OracleResult(
            executed=False,
            parser_changed=parser_hit,
            candidate=None,
            hooks=[],
            timeout=timeout,
            error=last_error,
        )


    def evaluate_document(self, html: str) -> OracleResult:
        row = self.check_snippet(html, request_id=0, as_document=True)
        executed = bool(row.get("executed"))
        return OracleResult(
            executed=executed,
            parser_changed=executed or parser_changed(html),
            candidate=html[:500] if executed else None,
            hooks=row.get("hooks") or [],
            error=row.get("error"),
        )


def evaluate_one(payload: str) -> OracleResult:
    with JsdomOracle() as oracle:
        return oracle.evaluate(payload)


def evaluate_many(payloads: Iterable[str]) -> list[OracleResult]:
    items = list(payloads)
    with JsdomOracle() as oracle:
        return [oracle.evaluate(p) for p in items]


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "<script>alert(1)</script>"
    result = evaluate_one(sample)
    print(
        json.dumps(
            {
                "executed": result.executed,
                "parser_changed": result.parser_changed,
                "candidate": result.candidate,
                "hooks": result.hooks,
                "error": result.error,
            },
            ensure_ascii=False,
        )
    )
