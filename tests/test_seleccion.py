"""Tests de las selecciones de demostraciones (src/data/seleccion.py): tramos."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from gymnasium import spaces

from src.data.demostraciones import validar_demostraciones
from src.data.seleccion import recortar_partida
from src.data.tipos import Trayectoria, comprobar_tramo

OBSERVACION = spaces.Box(-np.inf, np.inf, (4,), np.float32)
DISCRETA = spaces.Discrete(3)
CONTINUA = spaces.Box(-100.0, 100.0, (2,), np.float32)
PASOS = 5


def _partida(continua: bool = False, final: str = "terminado", pasos: int = PASOS,
             identificador: int = 7) -> Trayectoria:
    """Partida de prueba con valores distintos en cada paso."""
    estados = np.arange((pasos + 1) * 4, dtype=np.float32).reshape(pasos + 1, 4)
    acciones = (np.arange(pasos * 2, dtype=np.float32).reshape(pasos, 2) if continua
                else np.arange(pasos, dtype=np.int64) % 3)
    terminados = np.zeros(pasos, dtype=bool)
    truncados = np.zeros(pasos, dtype=bool)
    (terminados if final == "terminado" else truncados)[-1] = True
    return Trayectoria(identificador, estados[:-1], acciones, estados[1:],
                       np.arange(pasos, dtype=np.float32) + 0.5, terminados, truncados)


def _espacio(continua: bool) -> spaces.Space:
    return CONTINUA if continua else DISCRETA


# --- comprobar_tramo -------------------------------------------------------------------

@pytest.mark.parametrize("tramo", [[0, 1], (2, 5), [np.int64(1), np.int32(3)], [0, 10**12]])
def test_tramos_que_valen(tramo):
    assert comprobar_tramo(tramo) == (int(tramo[0]), int(tramo[1]))
    assert all(type(v) is int for v in comprobar_tramo(tramo))


@pytest.mark.parametrize("tramo", [[1, 1], [3, 2], [-1, 2], [0], [0, 1, 2], [0.0, 1],
                                   [0, 1.0], [False, True], ["0", "1"], "01", None, {0: 1},
                                   np.array([0, 1])])
def test_tramos_que_no_valen(tramo):
    with pytest.raises(ValueError, match=r"\[inicio, fin\) con 0 <= inicio < fin"):
        comprobar_tramo(tramo)


def test_el_tramo_tiene_que_caber_en_la_partida():
    assert comprobar_tramo([2, 5], 5) == (2, 5)
    with pytest.raises(ValueError, match=r"\[2, 6\) no cabe en una partida de 5 pasos"):
        comprobar_tramo([2, 6], 5)


# --- recortar_partida ------------------------------------------------------------------

TODOS_LOS_TRAMOS = [(i, f) for i in range(PASOS) for f in range(i + 1, PASOS + 1)]


@pytest.mark.parametrize("continua", [False, True], ids=["discreta", "continua"])
@pytest.mark.parametrize("final", ["terminado", "truncado"])
@pytest.mark.parametrize("inicio, fin", TODOS_LOS_TRAMOS)
def test_cada_tramo_de_una_partida(continua, final, inicio, fin):
    """Todos los tramos de una partida de 5 pasos (el dominio es finito: 15)."""
    partida = _partida(continua, final)
    recorte = recortar_partida(partida, [inicio, fin])
    assert recorte.id_episodio == partida.id_episodio
    assert recorte.numero_pasos == fin - inicio
    for campo in ("estados", "acciones", "siguientes_estados", "recompensas_originales"):
        np.testing.assert_array_equal(getattr(recorte, campo),
                                      getattr(partida, campo)[inicio:fin])
        assert getattr(recorte, campo).dtype == getattr(partida, campo).dtype
    # Solo el último paso finaliza: con sus marcas si es el final real; truncado si no.
    assert not recorte.terminados[:-1].any() and not recorte.truncados[:-1].any()
    if fin == PASOS:
        assert (recorte.terminados[-1], recorte.truncados[-1]) == (final == "terminado",
                                                                   final == "truncado")
    else:
        assert (recorte.terminados[-1], recorte.truncados[-1]) == (False, True)
    validar_demostraciones([recorte], OBSERVACION, _espacio(continua))


@pytest.mark.parametrize("inicio, fin", [(i, f) for i, f in itertools.product(range(-1, 7),
                                                                               range(-1, 8))
                                         if not 0 <= i < f <= PASOS])
def test_los_tramos_que_no_caben_se_rechazan(inicio, fin):
    with pytest.raises(ValueError, match="inicio < fin|no cabe"):
        recortar_partida(_partida(), [inicio, fin])


def test_el_recorte_es_una_copia():
    partida = _partida()
    originales = {c: getattr(partida, c).copy() for c in ("estados", "acciones", "truncados")}
    recorte = recortar_partida(partida, (1, 3))
    # Marcar el corte como truncado no toca la partida original.
    for campo, valor in originales.items():
        np.testing.assert_array_equal(getattr(partida, campo), valor)
    recorte.estados[:] = -1
    recorte.acciones[:] = 0
    recorte.truncados[:] = False
    for campo, valor in originales.items():
        np.testing.assert_array_equal(getattr(partida, campo), valor)


def test_tramos_seguidos_reconstruyen_la_partida():
    partida = _partida()
    trozos = [recortar_partida(partida, tramo) for tramo in ([0, 2], [2, 3], [3, 5])]
    for campo in ("estados", "acciones", "siguientes_estados", "recompensas_originales"):
        np.testing.assert_array_equal(np.concatenate([getattr(t, campo) for t in trozos]),
                                      getattr(partida, campo))


def test_solo_recorta_trayectorias():
    with pytest.raises(ValueError, match="Trayectoria"):
        recortar_partida({"estados": []}, [0, 1])
