"""Tests de las comparaciones en la API: el contrato (ComparacionCrear y lo que manda el
trabajo) y, más adelante, la ruta POST /api/comparaciones."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.app.contrato import ArgumentosComparacionApi, ComparacionCrear, ProgresoComparacionApi

ID = "a" * 12


# --- ComparacionCrear ------------------------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "aleatorio": True},
    {"entorno": "reacher", "modelos": [ID]},
    {"entorno": "acrobot", "modelos": [ID, "b" * 12], "aleatorio": True, "partidas": 1000,
     "vista": True},
    {"entorno": "halfcheetah", "modelos": [ID], "partidas": 1, "aleatorio": False},
])
def test_peticiones_que_valen(cuerpo):
    peticion = ComparacionCrear.model_validate(cuerpo)
    assert peticion.entorno.value == cuerpo["entorno"]
    assert peticion.modelos == cuerpo.get("modelos", [])
    assert peticion.aleatorio is cuerpo.get("aleatorio", False)
    assert peticion.partidas == cuerpo.get("partidas")
    assert peticion.vista is cuerpo.get("vista", False)


@pytest.mark.parametrize("cambio, mensaje", [
    ({"modelos": [ID, ID]}, "repetidos"),
    ({"modelos": [], "aleatorio": False}, "nada que comparar"),
    ({"modelos": [ID], "extra": 1}, "Extra inputs"),
    ({"entorno": "pong"}, "entorno"),
    ({"modelos": [""]}, "at least 1 character"),
    ({"modelos": ID}, "valid list"),
    ({"partidas": 0}, "greater than or equal to 1"),
    ({"partidas": -3}, "greater than or equal to 1"),
    ({"partidas": True}, "valid integer"),
    ({"partidas": "3"}, "valid integer"),
    ({"partidas": 2.0}, "valid integer"),
    ({"aleatorio": "true"}, "valid boolean"),
    ({"aleatorio": 1}, "valid boolean"),
    ({"vista": 1}, "valid boolean"),
])
def test_peticiones_que_no_valen(cambio, mensaje):
    cuerpo = {"entorno": "cartpole", "modelos": [ID], **cambio}
    with pytest.raises(ValidationError, match=mensaje):
        ComparacionCrear.model_validate(cuerpo)


def test_sin_entorno_no_vale():
    with pytest.raises(ValidationError, match="entorno"):
        ComparacionCrear.model_validate({"aleatorio": True})


# --- lo que manda el trabajo ----------------------------------------------------------------

def test_argumentos_publicos_con_partidas_y_semilla_efectivas():
    argumentos = {"entorno": "cartpole", "modelos": [ID], "aleatorio": True, "partidas": 50,
                  "semilla": 9000, "vista": False}
    validados = ArgumentosComparacionApi.model_validate(argumentos)
    assert validados.model_dump(mode="json") == argumentos
    for falta in ("partidas", "semilla"):
        with pytest.raises(ValidationError, match=falta):
            ArgumentosComparacionApi.model_validate(
                {k: v for k, v in argumentos.items() if k != falta})


@pytest.mark.parametrize("medidas, vale", [
    ({"modelo": ID, "partida": 0, "retorno": 9.5, "pasos": 10}, True),
    ({"modelo": "aleatorio", "partida": 3, "retorno": -200.0, "pasos": 200}, True),
    ({"modelo": ID, "partida": -1, "retorno": 1.0, "pasos": 1}, False),
    ({"modelo": ID, "partida": 0, "retorno": 1.0, "pasos": 0}, False),
    ({"modelo": ID, "partida": 0, "pasos": 1}, False),
])
def test_progreso_de_una_comparacion(medidas, vale):
    if vale:
        assert ProgresoComparacionApi.model_validate(medidas).modelo == medidas["modelo"]
    else:
        with pytest.raises(ValidationError):
            ProgresoComparacionApi.model_validate(medidas)
