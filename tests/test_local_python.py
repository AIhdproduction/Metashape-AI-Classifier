import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load():
    spec = importlib.util.spec_from_file_location("local_python", ROOT / "tools/local_python.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_network_base_python_is_copied_and_the_venv_repointed(tmp_path, monkeypatch):
    module = load()
    share = tmp_path / "share" / "python"
    (share / "Lib" / "site-packages" / "big").mkdir(parents=True)
    (share / "Lib" / "os.py").write_text("x", encoding="utf-8")
    (share / "DLLs").mkdir()
    (share / "python.exe").write_bytes(b"exe")
    config = tmp_path / ".venv" / "pyvenv.cfg"
    config.parent.mkdir()
    config.write_text(f"home = {share}\nversion = 3.12.0\nexecutable = {share}\python.exe\n")
    monkeypatch.setattr(module, "CONFIG", config)
    monkeypatch.setattr(module, "LOCAL", tmp_path / ".python-local")
    monkeypatch.setattr(module, "is_remote", lambda path: Path(path) == share)
    assert module.main() == 0
    local = tmp_path / ".python-local"
    assert (local / "Lib" / "os.py").is_file() and (local / "python.exe").is_file()
    # The installed packages stay in the venv; the copy never takes the share's packages.
    assert not (local / "Lib" / "site-packages" / "big").exists()
    text = config.read_text()
    assert f"home = {local}" in text and "version = 3.12.0" in text
    assert (config.parent / "pyvenv.cfg.network").read_text().startswith(f"home = {share}")
    # A second run changes nothing.
    stamp = (local / "source.txt").stat().st_mtime_ns
    assert module.main() == 0 and (local / "source.txt").stat().st_mtime_ns == stamp


def test_local_base_python_is_left_alone(tmp_path, monkeypatch):
    module = load()
    config = tmp_path / "pyvenv.cfg"
    config.write_text(f"home = {tmp_path}\n")
    monkeypatch.setattr(module, "CONFIG", config)
    monkeypatch.setattr(module, "LOCAL", tmp_path / ".python-local")
    monkeypatch.setattr(module, "is_remote", lambda path: False)
    assert module.main() == 0 and not (tmp_path / ".python-local").exists()


def test_unc_paths_count_as_network():
    module = load()
    assert module.is_remote(Path("//server/share/python"))
    assert not module.is_remote(ROOT)  # The repository is on a local disk here.
