"""noflash placeholder logic, without needing evo2/vortex installed."""
import sys
import types

import pytest

from marv_hyena import noflash


@pytest.fixture
def clean_modules(monkeypatch):
    monkeypatch.delitem(sys.modules, "flash_attn_2_cuda", raising=False)
    monkeypatch.delitem(sys.modules, "transformer_engine", raising=False)
    yield


def test_placeholder_is_not_reported_as_real_flash_attn(clean_modules, monkeypatch):
    stub = types.ModuleType("flash_attn_2_cuda")  # __spec__ is None, like the real placeholder
    setattr(stub, noflash._STUB_MARKER, True)
    monkeypatch.setitem(sys.modules, "flash_attn_2_cuda", stub)
    assert noflash.flash_attn_available() is False  # used to raise ValueError: __spec__ is None


def test_real_module_in_sys_modules_counts(clean_modules, monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    assert noflash.flash_attn_available() is True


def test_broken_transformer_engine_becomes_import_error(clean_modules, tmp_path, monkeypatch):
    pkg = tmp_path / "transformer_engine"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('Found empty `transformer-engine` meta package installed.')")
    monkeypatch.syspath_prepend(str(tmp_path))
    noflash.ignore_broken_transformer_engine()
    with pytest.raises(ImportError):
        import transformer_engine  # noqa: F401
