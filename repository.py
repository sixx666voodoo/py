"""Side load repository for managing extended audio data."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4


@dataclass
class SideLoadEntry:
    """Representation of a stored file inside the side load repository."""

    identifier: str
    original_name: str
    stored_name: str
    size_bytes: int
    created_at: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Serialize the entry so it can be persisted in ``manifest.json``."""

        return {
            "original_name": self.original_name,
            "stored_name": self.stored_name,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class SideLoadRepository:
    """Manage a workspace for side loaded files.

    The repository keeps a manifest that maps generated identifiers to the
    files stored in the ``files`` sub directory. Files can be resolved either by
    their identifier or by providing a file path that will be imported into the
    repository on demand.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.manifest_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._entries = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_file(
        self, source_path: Path | str, metadata: Optional[Dict[str, str]] = None
    ) -> SideLoadEntry:
        """Copy ``source_path`` into the repository and return the entry.

        ``source_path`` must exist on disk. The file is copied into the
        repository with a generated identifier and the manifest is updated.
        """

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        metadata = {key: str(value) for key, value in (metadata or {}).items()}
        identifier = self._generate_identifier(source.name)
        stored_name = f"{identifier}{source.suffix}"
        destination = self.files_dir / stored_name
        shutil.copy2(source, destination)
        created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        entry = SideLoadEntry(
            identifier=identifier,
            original_name=source.name,
            stored_name=stored_name,
            size_bytes=destination.stat().st_size,
            created_at=created_at,
            metadata=metadata,
        )
        self._entries[identifier] = entry
        self._save_manifest()
        return entry

    def resolve(
        self, reference: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Return a file path inside the repository for ``reference``.

        When ``reference`` refers to an existing file on disk the file is copied
        into the repository and the stored location is returned. Otherwise the
        reference is interpreted as an identifier that must already exist in the
        manifest.
        """

        candidate = Path(reference)
        if candidate.is_file():
            entry = self.store_file(candidate, metadata=metadata)
            return str(self.files_dir / entry.stored_name)
        entry = self.get_entry(reference)
        if entry is None:
            raise FileNotFoundError(
                f"Unable to resolve '{reference}'. Provide an existing file path "
                "or a known side load identifier."
            )
        return str(self.files_dir / entry.stored_name)

    def get_entry(self, identifier: str) -> Optional[SideLoadEntry]:
        return self._entries.get(identifier)

    def list_entries(self) -> List[SideLoadEntry]:
        return list(self._entries.values())

    def remove(self, identifier: str) -> None:
        entry = self._entries.pop(identifier, None)
        if entry is None:
            raise KeyError(identifier)
        stored_file = self.files_dir / entry.stored_name
        if stored_file.exists():
            stored_file.unlink()
        self._save_manifest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_manifest(self) -> Dict[str, SideLoadEntry]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries: Dict[str, SideLoadEntry] = {}
        for identifier, payload in data.items():
            entries[identifier] = SideLoadEntry(
                identifier=identifier,
                original_name=payload["original_name"],
                stored_name=payload["stored_name"],
                size_bytes=int(payload["size_bytes"]),
                created_at=payload["created_at"],
                metadata={
                    str(key): str(value) for key, value in payload.get("metadata", {}).items()
                },
            )
        return entries

    def _save_manifest(self) -> None:
        data = {identifier: entry.to_dict() for identifier, entry in self._entries.items()}
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _generate_identifier(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        base = slug or "entry"
        identifier = f"{base}-{uuid4().hex[:8]}"
        while identifier in self._entries:
            identifier = f"{base}-{uuid4().hex[:8]}"
        return identifier


def parse_metadata_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    """Convert an iterable of ``key=value`` strings into a metadata dictionary."""

    metadata: Dict[str, str] = {}
    for pair in pairs:
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid metadata entry '{pair}'. Use key=value format.")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Metadata keys must not be empty.")
        metadata[key] = value.strip()
    return metadata

