from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class FrameworkAdapter(ABC):
    name: str

    @abstractmethod
    def build_command(self, spec: Dict[str, Any], repo_root: Path) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def build_env(self, spec: Dict[str, Any]) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def default_cwd(self, repo_root: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def find_latest_raw_result(
        self,
        spec: Dict[str, Any],
        repo_root: Path,
        started_epoch: float,
    ) -> Optional[Path]:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw_payload: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _newest_matching(self, root: Path, pattern: str, started_epoch: float) -> Optional[Path]:
        candidates = []
        for p in root.glob(pattern):
            try:
                mtime = p.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime >= started_epoch - 2.0:
                candidates.append((mtime, p))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
