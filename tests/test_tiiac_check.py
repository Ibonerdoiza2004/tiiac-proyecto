"""Regresiones del comprobador para enlaces de W&B Cloud y Forge."""

from __future__ import annotations

import pytest

from tools.tiiac_check import check_tracking


@pytest.mark.parametrize("url", [
    "https://wandb.ai/equipo/proyecto",
    "https://api.wandb.ai/equipo/proyecto",
    "https://forge.coreweave.com/wandb/equipo/proyecto",
])
def test_check_tracking_reconoce_urls_de_proyecto_wandb(tmp_path, url):
    """La puntuación reconoce los host de W&B Cloud y Forge sin cambiar el requisito."""
    carpeta = tmp_path / "src"
    carpeta.mkdir()
    (carpeta / "tracking.py").write_text("import wandb\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(f"Proyecto: {url}\n", encoding="utf-8")

    resultado = check_tracking(tmp_path)

    assert resultado.pts == 10
    assert not resultado.mal


def test_check_tracking_no_acepta_un_enlace_ajeno(tmp_path):
    """Reconocer Forge no convierte cualquier URL en un enlace de seguimiento válido."""
    carpeta = tmp_path / "src"
    carpeta.mkdir()
    (carpeta / "tracking.py").write_text("import wandb\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Proyecto: https://example.com/proyecto\n",
                                        encoding="utf-8")

    resultado = check_tracking(tmp_path)

    assert resultado.pts == 5
    assert len(resultado.mal) == 1
