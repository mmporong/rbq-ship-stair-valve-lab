"""Download pinned third-party robot assets into .assets/ and verify their SHA-256."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "assets.lock.json"
SOURCES = {
    "menagerie": {
        "repo": "google-deepmind/mujoco_menagerie",
        "commit": "98a70c569c4c1b6da467a20012f766ee1f8dc521",
        "prefix": "unitree_go2/",
    },
    "z1": {
        "repo": "google-deepmind/mujoco_menagerie",
        "commit": "98a70c569c4c1b6da467a20012f766ee1f8dc521",
        "prefix": "unitree_z1/",
    },
    "rbq": {
        "repo": "RainbowRobotics/RBQ",
        "commit": "c7b80ba49f9a79f983cb977e2726cb2246701fcf",
        "prefix": "resources/",
    },
    "d1": {
        "repo": "JeewanthaSadaruwan/unitree-D1-550-Robot-ARM",
        "commit": "3bb0435cbf027826995b96248e60c2026e6e6be3",
        "prefix": "d1_550_description/",
    },
}
KEEP = {
    "menagerie": ("go2.xml", "assets/", "LICENSE"),
    "z1": ("z1_gripper.xml", "assets/", "LICENSE"),
    "rbq": (
        "model/rbq10_with_rb1.xml",
        "model/README.md",
        "meshes/rbq/trunk.obj",
        "meshes/rbq/hip2.obj",
        "meshes/rbq/hip3.obj",
        "meshes/rbq/thigh.obj",
        "meshes/rbq/calf.obj",
        "meshes/rb1/",
    ),
    "d1": ("urdf/d1_550_description.urdf", "meshes/"),
}


def raw_url(source: dict, path: str) -> str:
    return f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{path}"


def list_files(name: str) -> list[str]:
    source = SOURCES[name]
    output = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{source['repo']}/git/trees/{source['commit']}?recursive=1",
            "-q",
            '.tree[] | select(.type=="blob") | .path',
        ],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout.split("\n")
    prefix = source["prefix"]
    keep = tuple(prefix + item for item in KEEP[name])
    return sorted(path for path in output if path.startswith(keep))


def download(name: str, path: str) -> bytes:
    with urllib.request.urlopen(raw_url(SOURCES[name], path), timeout=60) as response:
        return response.read()


def write_lock() -> None:
    lock = {}
    for name, source in SOURCES.items():
        files = {
            path: hashlib.sha256(download(name, path)).hexdigest() for path in list_files(name)
        }
        lock[name] = {"repo": source["repo"], "commit": source["commit"], "files": files}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def fetch() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    count = 0
    for name, entry in lock.items():
        for path, digest in entry["files"].items():
            target = ROOT / ".assets" / name / path
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                continue
            data = download(name, path)
            actual = hashlib.sha256(data).hexdigest()
            if actual != digest:
                raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            count += 1
    print(f"assets ready, downloaded {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-lock", action="store_true")
    if parser.parse_args().write_lock:
        write_lock()
    fetch()
