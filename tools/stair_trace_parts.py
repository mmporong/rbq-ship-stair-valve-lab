"""Archive and restore one large mission.npz byte-for-byte in Git-sized parts."""

import argparse
import copy
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path

import numpy as np
from compare_stair_replays import LOAD_CHANNELS

PART_SIZE = 48 * 1024 * 1024


def validate_npy_member(source, size):
    version = np.lib.format.read_magic(source)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(source)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(source)
    else:
        raise ValueError(f"Unsupported NPY format version {version}")
    if dtype.hasobject:
        raise ValueError("Object dtypes cannot be restored without loading")
    if source.tell() + math.prod(shape) * dtype.itemsize != size:
        raise ValueError("NPY member size disagrees with its array header")
    # Reading through the member also validates its ZIP checksum.
    while source.read(8 * 1024**2):
        pass


def copy_record(source, destination, length):
    while length:
        data = source.read(min(8 * 1024**2, length))
        if not data:
            raise ValueError("Truncated ZIP member in source recording")
        destination.write(data)
        length -= len(data)


def relocated_info(member, new_offset):
    info = copy.copy(member)
    if new_offset != member.header_offset and info.extra:
        # ZipFile regenerates ZIP64 sizes and offsets for the new central directory.
        extra = member.extra
        keep = bytearray()
        while extra:
            if len(extra) < 4:
                raise ValueError("Truncated ZIP extra field")
            tag, size = struct.unpack_from("<HH", extra)
            end = 4 + size
            if end > len(extra):
                raise ValueError("Truncated ZIP extra field payload")
            if tag != 1:
                keep.extend(extra[:end])
            extra = extra[end:]
        info.extra = bytes(keep)
    info.header_offset = new_offset
    return info


def pack(source, directory, part_size=PART_SIZE):
    if part_size < 1:
        raise ValueError("Part size must be positive")
    directory.mkdir(parents=True, exist_ok=False)
    total = hashlib.sha256()
    parts = []
    with source.open("rb") as stream:
        while data := stream.read(part_size):
            name = f"mission.npz.{len(parts) + 1:03d}"
            with (directory / name).open("xb") as dest:
                dest.write(data)
            total.update(data)
            parts.append(
                {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
    if not parts:
        raise ValueError("The source trace is empty")
    manifest = {
        "name": "mission.npz",
        "bytes": sum(p["bytes"] for p in parts),
        "sha256": total.hexdigest(),
        "parts": parts,
    }
    with (directory / "parts.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    return manifest


def restore(directory, destination):
    manifest = json.loads((directory / "parts.json").read_bytes())
    parts = manifest["parts"]
    if not parts or manifest["name"] != "mission.npz":
        raise ValueError("Invalid mission archive")
    if destination.exists():
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == manifest["sha256"]:
                return manifest
        raise FileExistsError(f"Different file already exists at {destination}")
    total = hashlib.sha256()
    length = 0
    with destination.open("xb") as output:
        try:
            for i, part in enumerate(parts, start=1):
                name = f"mission.npz.{i:03d}"
                if part["name"] != name:
                    raise ValueError(f"Expected part {name}")
                data = (directory / name).read_bytes()
                if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                    raise ValueError(f"Damaged part {name}")
                output.write(data)
                length += len(data)
                total.update(data)
            if length != manifest["bytes"] or total.hexdigest() != manifest["sha256"]:
                raise ValueError("Reconstructed trace does not match the whole-file digest")
        except BaseException:
            output.close()
            destination.unlink()
            raise
    return manifest


def restore_prefix(source, destination, samples, expected_sha256):
    """Recover an independently recorded prefix and check its original NPZ fingerprint."""
    with np.load(source, allow_pickle=False) as mission:
        total = len(mission["time_s"])
        if not 0 < samples <= total:
            raise ValueError("Prefix sample count must lie inside the source recording")
        with destination.open("x+b") as output:
            try:
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for key in mission.files:
                        channel = mission[key]
                        if channel.ndim == 0 or len(channel) != total:
                            raise ValueError(f"{key} does not share the mission time axis")
                        window = channel[:samples]
                        if channel.flags.f_contiguous and not channel.flags.c_contiguous:
                            window = np.asfortranarray(window)
                        info = copy.copy(mission.zip.getinfo(key + ".npy"))
                        with archive.open(info, "w", force_zip64=True) as member:
                            np.lib.format.write_array(member, window, allow_pickle=False)
                output.seek(0)
                if hashlib.file_digest(output, "sha256").hexdigest() != expected_sha256:
                    raise ValueError("Reconstructed prefix differs from its independent recording")
            except BaseException:
                output.close()
                destination.unlink()
                raise


def restore_without_loads(source, destination, expected_sha256):
    """Remove the complete structural-load sensor group and verify the original lean recording."""
    try:
        bytes.fromhex(expected_sha256)
    except ValueError as error:
        raise ValueError("Expected SHA-256 must be hexadecimal") from error
    if len(expected_sha256) != 64:
        raise ValueError("Expected SHA-256 must contain 64 hexadecimal characters")
    if destination.exists():
        raise FileExistsError(f"Output already exists at {destination}")

    with zipfile.ZipFile(source, "r") as recorded:
        members = recorded.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names) or any(not name.endswith(".npy") for name in names):
            raise ValueError("Source is not a canonical NPZ recording")
        channels = {name.removesuffix(".npy") for name in names}
        missing = LOAD_CHANNELS - channels
        if missing:
            raise ValueError(f"Source is missing load channels: {sorted(missing)}")
        retained = [
            member
            for member in members
            if member.filename.removesuffix(".npy") not in LOAD_CHANNELS
        ]
        if not retained:
            raise ValueError("Removing load channels would leave an empty recording")

        offsets = sorted(member.header_offset for member in members)
        ends = dict(zip(offsets, offsets[1:] + [recorded.start_dir], strict=True))
        if len(ends) != len(members) or offsets[0] < 0 or offsets[-1] >= recorded.start_dir:
            raise ValueError("Invalid ZIP member offsets")
        with destination.open("x+b") as output:
            try:
                with source.open("rb") as original, zipfile.ZipFile(output, "w") as lean:
                    lean.comment = recorded.comment
                    for member in retained:
                        with recorded.open(member) as array:
                            validate_npy_member(array, member.file_size)
                        info = relocated_info(member, output.tell())
                        original.seek(member.header_offset)
                        length = ends[member.header_offset] - member.header_offset
                        copy_record(original, output, length)
                        # Reuse the compressed local record. ZipFile writes the directory on close.
                        lean.filelist.append(info)
                    lean.start_dir = output.tell()
                output.seek(0)
                actual = hashlib.file_digest(output, "sha256").hexdigest()
                if actual != expected_sha256.lower():
                    raise ValueError(
                        "Recording without load channels differs from its independent original"
                    )
            except BaseException:
                output.close()
                destination.unlink()
                raise
    return {
        "source": source,
        "destination": destination,
        "removed_channels": sorted(LOAD_CHANNELS),
        "sha256": actual,
        "verified_against_independent_original": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    split = sub.add_parser("pack")
    split.add_argument("source", type=Path)
    split.add_argument("directory", type=Path)
    join = sub.add_parser("restore")
    join.add_argument("directory", type=Path)
    join.add_argument("destination", type=Path)
    prefix = sub.add_parser("restore-prefix")
    prefix.add_argument("source", type=Path)
    prefix.add_argument("destination", type=Path)
    prefix.add_argument("--samples", type=int, required=True)
    prefix.add_argument("--sha256", required=True, help="digest of the original short recording")
    lean = sub.add_parser("restore-without-loads")
    lean.add_argument(
        "source", type=Path, help="recording that contains all structural load channels"
    )
    lean.add_argument("destination", type=Path)
    lean.add_argument(
        "--sha256", required=True, help="digest of the independently recorded lean NPZ"
    )
    args = parser.parse_args()
    if args.action == "restore-prefix":
        restore_prefix(args.source, args.destination, args.samples, args.sha256)
        print(f"{args.samples} samples, SHA-256 {args.sha256}")
        return
    if args.action == "restore-without-loads":
        result = restore_without_loads(args.source, args.destination, args.sha256)
        print(
            f"removed {len(result['removed_channels'])} load channels, "
            f"matches independently recorded SHA-256 {result['sha256']}"
        )
        return
    if args.action == "pack":
        result = pack(args.source, args.directory)
    else:
        result = restore(args.directory, args.destination)
    print(f"{result['bytes']} bytes, {len(result['parts'])} parts, SHA-256 {result['sha256']}")


if __name__ == "__main__":
    main()
