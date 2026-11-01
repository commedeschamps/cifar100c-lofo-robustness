"""Explicit official-only download and safe, checksum-verified extraction."""
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from ..utils import save_json
from .data import ARCHIVE_MD5, CORRUPTIONS, SOURCE, URL, file_hash


def checked_members(archive):
    members = archive.getmembers()
    seen = set()
    for member in members:
        p = PurePosixPath(member.name)
        if (p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "CIFAR-100-C"
                or not (member.isdir() or member.isfile()) or member.name in seen):
            raise ValueError(f"Unsafe or duplicate archive member: {member.name}")
        seen.add(member.name)
    required = {f"CIFAR-100-C/{n}.npy" for n in (*CORRUPTIONS, "labels")}
    if not required.issubset({m.name for m in members if m.isfile()}):
        raise ValueError("Official archive is missing required benchmark files.")
    return members


def verify_archive(archive_path, root):
    archive_path, root = Path(archive_path), Path(root)
    if file_hash(archive_path, "md5") != ARCHIVE_MD5:
        raise ValueError("Archive MD5 differs from official Zenodo CIFAR-100-C; refusing it.")
    records = {}
    with tarfile.open(archive_path, "r:") as archive:
        for member in checked_members(archive):
            if member.isfile() and member.name.endswith(".npy"):
                relative = PurePosixPath(member.name).relative_to("CIFAR-100-C")
                with archive.extractfile(member) as stream:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                        h.update(chunk)
                path = root / str(relative)
                if not path.is_file() or file_hash(path) != h.hexdigest():
                    raise ValueError(f"Extracted file differs from verified archive: {path}")
                records[str(relative)] = h.hexdigest()
    receipt = {"source": SOURCE, "download_url": URL, "archive_md5": ARCHIVE_MD5,
               "archive_sha256": file_hash(archive_path), "files": records}
    save_json(root / "official_provenance.json", receipt)
    return receipt


def download(data_root):
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "CIFAR-100-C.tar"
    target = root / "CIFAR-100-C"
    if target.exists():
        raise FileExistsError(f"Will not overwrite {target}; use verify with the existing official archive.")
    # 2.9 GB tar plus ~2.9 GB extracted, with working-space margin.
    if shutil.disk_usage(root).free < 7_000_000_000:
        raise OSError("Need at least 7 GB free for official archive, extraction, and working space.")
    if not archive_path.exists():
        partial = archive_path.with_suffix(".tar.part")
        print(f"Downloading only the official source: {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=8*1024*1024)
        if file_hash(partial, "md5") != ARCHIVE_MD5:
            raise ValueError(f"Official archive checksum failed; retained for diagnosis: {partial}")
        partial.replace(archive_path)
    if file_hash(archive_path, "md5") != ARCHIVE_MD5:
        raise ValueError("Existing archive fails official MD5. No extraction performed.")
    with tempfile.TemporaryDirectory(prefix="cifar100c-extract-", dir=root) as temporary:
        with tarfile.open(archive_path, "r:") as archive:
            members = checked_members(archive)
            if shutil.disk_usage(root).free < sum(m.size for m in members) + 500_000_000:
                raise OSError("Insufficient extraction space.")
            archive.extractall(temporary, members=members, filter="data")
        extracted = Path(temporary) / "CIFAR-100-C"
        verify_archive(archive_path, extracted)
        extracted.replace(target)
    print(f"Verified official benchmark ready at {target}")
