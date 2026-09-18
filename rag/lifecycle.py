import hashlib
import json
from pathlib import Path

from rag import chunk, embeddings, ingest, store


def scan_folder(folder_path: str) -> dict[str, str]:
    folder = Path(folder_path)
    if not folder.is_dir():
        return {}
    manifest = {}
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ingest.SUPPORTED_EXTENSIONS:
            manifest[file_path.name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return manifest


def load_manifest(manifest_path: str) -> dict[str, str]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(manifest_path: str, manifest: dict[str, str]) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def sync(collection, folder_path: str, manifest_path: str) -> dict[str, list[str]]:
    current = scan_folder(folder_path)
    previous = load_manifest(manifest_path)

    added = [name for name in current if name not in previous]
    deleted = [name for name in previous if name not in current]
    modified = [name for name in current if name in previous and current[name] != previous[name]]

    for name in deleted + modified:
        store.delete_by_source(collection, name)

    for name in added + modified:
        document = ingest.load_document(str(Path(folder_path) / name))
        chunks = chunk.chunk_text(document.text)
        if chunks:
            vectors = embeddings.embed_texts(chunks)
            store.add_chunks(collection, document.source, chunks, vectors)

    save_manifest(manifest_path, current)
    return {"added": added, "deleted": deleted, "modified": modified}
