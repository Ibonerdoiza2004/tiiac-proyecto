"""Tests de src/utils/archivos.py que no están en los de cada guardado: el cerrojo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.archivos import turno_exclusivo

# Intenta coger el cerrojo sin esperar: sale con 0 si puede y con 1 si otro lo tiene.
INTENTAR = """
import fcntl, sys
with open(sys.argv[1], "a") as archivo:
    try:
        fcntl.flock(archivo, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
"""


def _otro_proceso_puede(cerrojo: Path) -> bool:
    return subprocess.run([sys.executable, "-c", INTENTAR, str(cerrojo)],
                          timeout=30).returncode == 0


def test_mientras_dura_el_turno_otro_proceso_espera(tmp_path):
    cerrojo = tmp_path / "carpeta" / ".algo.lock"
    with turno_exclusivo(cerrojo):
        assert cerrojo.is_file()  # crea el archivo y su carpeta
        assert not _otro_proceso_puede(cerrojo)
    assert _otro_proceso_puede(cerrojo)


def test_si_falla_dentro_suelta_el_cerrojo(tmp_path):
    cerrojo = tmp_path / ".algo.lock"
    try:
        with turno_exclusivo(cerrojo):
            raise RuntimeError("falla")
    except RuntimeError:
        pass
    assert _otro_proceso_puede(cerrojo)


def test_sin_esperar_falla_si_otro_tiene_el_cerrojo(tmp_path):
    cerrojo = tmp_path / ".algo.lock"
    with turno_exclusivo(cerrojo, esperar=False):
        assert not _otro_proceso_puede(cerrojo)
        with pytest.raises(BlockingIOError):  # también desde el mismo proceso
            with turno_exclusivo(cerrojo, esperar=False):
                pass
    with turno_exclusivo(cerrojo, esperar=False):  # libre: entra
        pass
    assert _otro_proceso_puede(cerrojo)
