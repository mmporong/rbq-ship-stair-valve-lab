import hashlib
import zipfile

import numpy as np
import pytest

LOAD_CHANNELS = {
    "joint_force_world_n",
    "joint_moment_world_nm",
    "joint_off_axis_bending_nm",
    "joint_reaction_n",
    "rb1_mount_wrench_world",
    "foot_surface_force_world_n",
}


def test_restore_prefix_preserves_column_order_and_original_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_trace_parts import restore_prefix

    source = tmp_path / "full.npz"
    original = tmp_path / "original.npz"
    recovered = tmp_path / "recovered.npz"
    feet = np.asfortranarray(np.arange(24, dtype=float).reshape(6, 4))
    feet[0, 0] = -0.0
    np.savez_compressed(source, time_s=np.arange(6), feet=feet)
    np.savez_compressed(original, time_s=np.arange(3), feet=np.asfortranarray(feet[:3]))
    expected = hashlib.sha256(original.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="differs from its independent recording"):
        restore_prefix(source, recovered, 3, "0" * 64)
    assert not recovered.exists()
    restore_prefix(source, recovered, 3, expected)
    assert original.read_bytes() == recovered.read_bytes()
    with pytest.raises(FileExistsError):
        restore_prefix(source, recovered, 3, expected)


def test_restore_without_loads_reproduces_independent_npz_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_trace_parts import restore_without_loads

    loaded = tmp_path / "loaded.npz"
    original = tmp_path / "lean.npz"
    restored = tmp_path / "restored.npz"
    signed = np.array([-0.0, 0.0, 1.0])
    fortran = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4))
    beyond_write_chunk = np.arange(16 * 1024**2 // 8 + 3, dtype=np.float64)
    common = {
        "time_s": signed,
        "fortran": fortran,
        "beyond_write_chunk": beyond_write_chunk,
        "stage": np.array(["a", "b", "c"]),
    }
    sensors = {
        name: np.asfortranarray(np.arange(12.0).reshape(3, 4) + offset)
        for offset, name in enumerate(sorted(LOAD_CHANNELS))
    }
    np.savez_compressed(loaded, **common, **sensors)
    np.savez_compressed(original, **common)
    expected = hashlib.sha256(original.read_bytes()).hexdigest()

    report = restore_without_loads(loaded, restored, expected)

    assert report["verified_against_independent_original"]
    assert report["removed_channels"] == sorted(LOAD_CHANNELS)
    assert restored.read_bytes() == original.read_bytes()
    with np.load(restored, allow_pickle=False) as result:
        assert result.files == list(common)
        assert result["fortran"].flags.f_contiguous
        assert np.signbit(result["time_s"][0])
    with pytest.raises(FileExistsError):
        restore_without_loads(loaded, restored, expected)


@pytest.mark.parametrize("zip64_limit", [None, 256])
def test_restore_copies_interleaved_compressed_members(tmp_path, monkeypatch, zip64_limit):
    monkeypatch.syspath_prepend("tools")
    from stair_trace_parts import restore_without_loads

    if zip64_limit is not None:
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", zip64_limit)
    loaded, original, restored = (tmp_path / name for name in ("all.npz", "lean.npz", "out.npz"))
    samples = np.arange(16000, dtype=np.float64)
    sensors = sorted(LOAD_CHANNELS)
    names = [sensors[0], "time_s", *sensors[1:], "qpos"]
    for archive_path, keys in ((loaded, names), (original, ["time_s", "qpos"])):
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.comment = b"trace"
            for key in keys:
                with archive.open(key + ".npy", "w", force_zip64=True) as member:
                    target = samples if key in {"time_s", "qpos"} else samples[:100]
                    np.lib.format.write_array(member, target, allow_pickle=False)
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    restore_without_loads(loaded, restored, expected)
    assert restored.read_bytes() == original.read_bytes()


def test_restore_rejects_object_arrays_and_corrupted_member_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_trace_parts import restore_without_loads

    loaded, restored = tmp_path / "all.npz", tmp_path / "out.npz"
    sensors = {name: [0.0] for name in sorted(LOAD_CHANNELS)}
    np.savez(loaded, sample=np.array([None], dtype=object), **sensors)
    with pytest.raises(ValueError, match="Object dtypes"):
        restore_without_loads(loaded, restored, "0" * 64)
    assert not restored.exists()

    np.savez(loaded, sample=np.arange(2000), **sensors)
    with zipfile.ZipFile(loaded) as archive:
        last_byte = archive.infolist()[1].header_offset - 1
    with loaded.open("r+b") as stream:
        stream.seek(last_byte)
        changed = bytes([stream.read(1)[0] ^ 1])
        stream.seek(last_byte)
        stream.write(changed)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        restore_without_loads(loaded, restored, "0" * 64)
    assert not restored.exists()


def test_restore_without_loads_rejects_missing_channel_and_wrong_digest(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_trace_parts import restore_without_loads

    incomplete = tmp_path / "incomplete.npz"
    destination = tmp_path / "destination.npz"
    sensors = {name: [1.0] for name in sorted(LOAD_CHANNELS)[:-1]}
    np.savez_compressed(incomplete, time_s=[0.001], **sensors)
    with pytest.raises(ValueError, match="missing load channels"):
        restore_without_loads(incomplete, destination, "0" * 64)
    assert not destination.exists()

    complete = tmp_path / "complete.npz"
    np.savez_compressed(
        complete,
        time_s=[0.001],
        **{name: [1.0] for name in sorted(LOAD_CHANNELS)},
    )
    with pytest.raises(ValueError, match="differs from its independent original"):
        restore_without_loads(complete, destination, "0" * 64)
    assert not destination.exists()
