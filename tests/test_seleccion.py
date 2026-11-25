"""Tests de las selecciones de demostraciones (src/data/seleccion.py): subconjuntos."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from src.app.contrato import DemostracionApi
from src.app.rutas_biblioteca import como_demostracion
from src.data.biblioteca import (ARCHIVO_SUBCONJUNTO, Biblioteca, agregar, agregar_nuevo,
                                 crear_ficha, nuevo_id)
from src.data.demostraciones import guardar_demostraciones, resumen_demostraciones
from src.data.seleccion import (FORMATO_SUBCONJUNTO, VERSION_SUBCONJUNTO, cargar_subconjunto,
                                comprobar_subconjunto, crear_subconjunto)
from src.data.splits import huella_archivo
from src.data.tipos import Trayectoria

OBSERVACION = spaces.Box(-np.inf, np.inf, (4,), np.float32)
DISCRETA = spaces.Discrete(3)
PASOS = 5


def _partida(pasos: int = PASOS, identificador: int = 7) -> Trayectoria:
    """Partida de prueba con valores distintos en cada paso, terminada en el último."""
    estados = np.arange((pasos + 1) * 4, dtype=np.float32).reshape(pasos + 1, 4)
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    return Trayectoria(identificador, estados[:-1], np.arange(pasos, dtype=np.int64) % 3,
                       estados[1:], np.arange(pasos, dtype=np.float32) + 0.5, terminados,
                       np.zeros(pasos, dtype=bool))


# --- subconjuntos ----------------------------------------------------------------------

ENTORNOS = {"cartpole": "CartPole-v1", "acrobot": "Acrobot-v1"}
# Partidas del conjunto de prueba: id → pasos.
PASOS_CONJUNTO = {0: 4, 3: 6, 5: 5}
ELEGIDAS = [5, 0]


def _metadatos_experto(gym: str = "CartPole-v1") -> dict:
    return {"entorno": gym, "experto": "x", "procedencia": {}, "configuracion_experto": {}}


def _guardar_conjunto(ruta: Path, pasos: dict[int, int] = PASOS_CONJUNTO,
                      gym: str = "CartPole-v1") -> None:
    guardar_demostraciones(ruta, [_partida(pasos=p, identificador=i) for i, p in pasos.items()],
                           _metadatos_experto(gym), OBSERVACION, DISCRETA)


def _anadir_conjunto(biblioteca: Biblioteca, entorno: str = "cartpole", **extra) -> str:
    return agregar_nuevo(biblioteca.raiz, entorno, "demostraciones", "Conjunto", "dashboard",
                         lambda c: _guardar_conjunto(c / "demostraciones.npz",
                                                     gym=ENTORNOS[entorno]), **extra)["id"]


def _anadir_a_mano(biblioteca: Biblioteca, datos: object, entorno: str = "cartpole",
                   **extra) -> str:
    """Añade un subconjunto con el contenido que se quiera, sin comprobarlo."""
    ficha = crear_ficha(nuevo_id(), entorno, "demostraciones", "A mano", "dashboard", **extra)
    texto = datos if isinstance(datos, str) else json.dumps(datos)
    agregar(biblioteca.raiz, ficha,
            lambda c: (c / ARCHIVO_SUBCONJUNTO).write_text(texto, encoding="utf-8"))
    return ficha["id"]


def _datos(conjunto: str, huella: str, **cambios) -> dict:
    return {"formato": FORMATO_SUBCONJUNTO, "version": VERSION_SUBCONJUNTO, "conjunto": conjunto,
            "huella_dataset": huella, "partidas": [0], **cambios}


@pytest.fixture
def biblioteca(tmp_path: Path) -> Biblioteca:
    return Biblioteca(tmp_path / "biblioteca", ENTORNOS)


def test_crear_un_subconjunto_y_verlo_en_la_biblioteca(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    ficha = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "Mis mejores partidas")
    assert (ficha["origen"], ficha["fuente"], ficha["tipo"]) == ("dashboard", "experto",
                                                                 "demostraciones")
    carpeta = biblioteca.ruta(ficha["id"])
    assert not (carpeta / "demostraciones.npz").exists()  # no copia datos
    datos = json.loads((carpeta / ARCHIVO_SUBCONJUNTO).read_text(encoding="utf-8"))
    assert datos == {"formato": FORMATO_SUBCONJUNTO, "version": VERSION_SUBCONJUNTO,
                     "conjunto": conjunto,
                     "huella_dataset": huella_archivo(biblioteca.ruta(conjunto)
                                                      / "demostraciones.npz"),
                     "partidas": [0, 5]}  # en orden, sea cual sea el de la petición
    elemento = biblioteca.obtener(ficha["id"])
    assert elemento["valido"], elemento["motivo"]
    assert elemento["detalles"] == {"tipo": "subconjunto", "conjunto": conjunto, "partidas": 2,
                                    "pares": 4 + 5, "fuente": "experto", "splits": False}
    api = DemostracionApi.model_validate(como_demostracion(elemento))
    assert (api.tipo, api.partidas, api.pares, api.nombre) == ("subconjunto", 2, 9,
                                                               "Mis mejores partidas")


def test_el_mismo_subconjunto_da_el_mismo_archivo(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    contenidos = set()
    for partidas in ([5, 0], [0, 5], (np.int64(5), np.int32(0))):
        ficha = crear_subconjunto(biblioteca, conjunto, partidas, "s")
        contenidos.add((biblioteca.ruta(ficha["id"]) / ARCHIVO_SUBCONJUNTO).read_bytes())
    assert len(contenidos) == 1


def test_cargar_un_subconjunto_da_sus_partidas_enteras(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    ficha = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")
    partidas, datos = cargar_subconjunto(biblioteca, ficha["id"], OBSERVACION, DISCRETA)
    assert [p.id_episodio for p in partidas] == [0, 5] and datos["conjunto"] == conjunto
    for partida in partidas:
        esperada = _partida(pasos=PASOS_CONJUNTO[partida.id_episodio],
                            identificador=partida.id_episodio)
        for campo in ("estados", "acciones", "siguientes_estados", "terminados", "truncados"):
            np.testing.assert_array_equal(getattr(partida, campo), getattr(esperada, campo))


def test_la_fuente_del_subconjunto_es_la_de_su_conjunto(biblioteca):
    conjunto = _anadir_conjunto(biblioteca, fuente="experto")
    ficha = crear_subconjunto(biblioteca, conjunto, [3], "s")
    assert ficha["fuente"] == biblioteca.obtener(ficha["id"])["fuente"] == "experto"


@pytest.mark.parametrize("partidas, mensaje", [
    ([], "al menos una partida"),
    ({0: 1}, "lista de id_episodio"),
    ("05", "lista de id_episodio"),
    (5, "lista de id_episodio"),
    ([True], "no es un id de partida"),
    (["0"], "no es un id de partida"),
    ([1.0], "no es un id de partida"),
    ([0, 3, 0], "repetidas"),
    ([0, 9, 7], "las partidas [7, 9] no están en el conjunto"),
])
def test_crear_rechaza_partidas_que_no_valen(biblioteca, partidas, mensaje):
    conjunto = _anadir_conjunto(biblioteca)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        crear_subconjunto(biblioteca, conjunto, partidas, "s")
    assert [e["id"] for e in biblioteca.listar()] == [conjunto]  # no deja nada a medias


def test_crear_rechaza_lo_que_no_es_un_conjunto_valido(biblioteca):
    with pytest.raises(ValueError, match="No existe el conjunto"):
        crear_subconjunto(biblioteca, "a" * 12, ELEGIDAS, "s")
    conjunto = _anadir_conjunto(biblioteca)
    sub = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")["id"]
    with pytest.raises(ValueError, match="no es un conjunto de demostraciones"):
        crear_subconjunto(biblioteca, sub, ELEGIDAS, "s")

    def escribir_experto(carpeta: Path) -> None:
        (carpeta / "modelo.zip").write_bytes(b"x")
        (carpeta / "experto.json").write_text("{}")

    experto = agregar_nuevo(biblioteca.raiz, "cartpole", "experto", "e", "dashboard",
                            escribir_experto)["id"]
    with pytest.raises(ValueError, match="no es un conjunto de demostraciones"):
        crear_subconjunto(biblioteca, experto, ELEGIDAS, "s")
    roto = agregar_nuevo(biblioteca.raiz, "cartpole", "demostraciones", "roto", "dashboard",
                         lambda c: (c / "demostraciones.npz").write_bytes(b"no es un npz"))["id"]
    with pytest.raises(ValueError, match="no es válido"):
        crear_subconjunto(biblioteca, roto, ELEGIDAS, "s")


def test_sin_su_conjunto_o_si_ha_cambiado_deja_de_ser_valido(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    sub = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")["id"]
    npz = biblioteca.ruta(conjunto) / "demostraciones.npz"
    # Otro NPZ válido en su lugar: el conjunto sigue valiendo, el subconjunto no.
    npz.unlink()
    _guardar_conjunto(npz, {0: 4, 3: 6, 5: 7})
    assert biblioteca.obtener(conjunto)["valido"]
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"] and "el conjunto ha cambiado" in elemento["motivo"]
    assert elemento["detalles"] == {"tipo": "subconjunto"}
    assert como_demostracion(elemento)["tipo"] == "subconjunto"
    with pytest.raises(ValueError, match="no es válido: el conjunto ha cambiado"):
        cargar_subconjunto(biblioteca, sub, OBSERVACION, DISCRETA)
    shutil.rmtree(biblioteca.ruta(conjunto))
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"] and elemento["motivo"] == f"no existe su conjunto {conjunto}"


def test_un_subconjunto_con_partidas_que_su_conjunto_no_tiene_no_es_valido(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    huella = huella_archivo(biblioteca.ruta(conjunto) / "demostraciones.npz")
    sub = _anadir_a_mano(biblioteca, _datos(conjunto, huella, partidas=[0, 4]))
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"] and "las partidas [4] no están" in elemento["motivo"]


def test_cargar_comprueba_otra_vez_el_npz(biblioteca, monkeypatch):
    """Si el NPZ cambia entre listar y cargar, se nota al cargar."""
    from src.data import seleccion
    conjunto = _anadir_conjunto(biblioteca)
    sub = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")["id"]
    monkeypatch.setattr(seleccion, "huella_archivo", lambda ruta: "cd" * 32)
    with pytest.raises(ValueError, match="el conjunto ha cambiado"):
        cargar_subconjunto(biblioteca, sub, OBSERVACION, DISCRETA)


def test_subconjuntos_mal_apuntados_no_son_validos_y_no_hay_bucles(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    huella = huella_archivo(biblioteca.ruta(conjunto) / "demostraciones.npz")
    bueno = _anadir_a_mano(biblioteca, _datos(conjunto, huella))
    assert biblioteca.obtener(bueno)["valido"]
    otro_entorno = _anadir_conjunto(biblioteca, "acrobot")
    casos = {
        _anadir_a_mano(biblioteca, _datos(bueno, huella)):
            f"{bueno} es un subconjunto: solo se saca de un conjunto",
        _anadir_a_mano(biblioteca, _datos(otro_entorno, huella)):
            f"{otro_entorno} no son demostraciones de cartpole",
        _anadir_a_mano(biblioteca, "{no json"): "no se pudo leer subconjunto.json",
        _anadir_a_mano(biblioteca, _datos(conjunto, huella, version=2)):
            "Formato de subconjunto desconocido",
        _anadir_a_mano(biblioteca, _datos(conjunto, huella), fuente="humano"):
            "la ficha dice fuente 'humano' y su conjunto, 'experto'",
    }
    for id_sub, motivo in casos.items():
        elemento = biblioteca.obtener(id_sub)
        assert not elemento["valido"] and motivo in elemento["motivo"], (id_sub, elemento)
    # Uno que apunta a sí mismo no deja la lectura en bucle.
    a_si_mismo = crear_ficha(nuevo_id(), "cartpole", "demostraciones", "yo", "dashboard")
    agregar(biblioteca.raiz, a_si_mismo, lambda c: (c / ARCHIVO_SUBCONJUNTO).write_text(
        json.dumps(_datos(a_si_mismo["id"], huella))))
    assert "es un subconjunto" in biblioteca.obtener(a_si_mismo["id"])["motivo"]
    assert len(biblioteca.listar()) == 9


def test_cargar_rechaza_lo_que_no_es_un_subconjunto_valido(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    with pytest.raises(ValueError, match="no es un subconjunto"):
        cargar_subconjunto(biblioteca, conjunto, OBSERVACION, DISCRETA)
    with pytest.raises(ValueError, match="No existe el subconjunto"):
        cargar_subconjunto(biblioteca, "b" * 12, OBSERVACION, DISCRETA)


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda d: d.pop("partidas"), "exactamente"),
    (lambda d: d.update(otra=1), "exactamente"),
    (lambda d: d.update(formato="otro"), "Formato de subconjunto desconocido"),
    (lambda d: d.update(version=2), "Formato de subconjunto desconocido"),
    (lambda d: d.update(conjunto="../a"), "no es un id de la biblioteca"),
    (lambda d: d.update(huella_dataset="corta"), "huella"),
    (lambda d: d.update(partidas="todas"), "al menos una partida"),
    (lambda d: d.update(partidas=[]), "al menos una partida"),
    (lambda d: d.update(partidas=[{"id_episodio": 0, "tramo": [0, 1]}]),
     "no es un id de partida"),
    (lambda d: d.update(partidas=[2, 2]), "repetidas"),
])
def test_forma_del_subconjunto(cambio, mensaje):
    datos = _datos("a" * 12, "ab" * 32)
    assert comprobar_subconjunto(datos) == datos
    cambio(datos)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        comprobar_subconjunto(datos)


def test_el_resumen_da_los_pasos_de_cada_partida(tmp_path):
    _guardar_conjunto(tmp_path / "d.npz")
    assert resumen_demostraciones(tmp_path / "d.npz")["pasos_por_partida"] == PASOS_CONJUNTO


def test_si_su_conjunto_no_es_valido_el_subconjunto_tampoco(biblioteca):
    """Aunque su NPZ se lea bien: aquí la ficha del conjunto dice otra fuente."""
    conjunto = _anadir_conjunto(biblioteca, fuente="humano")
    assert not biblioteca.obtener(conjunto)["valido"]
    huella = huella_archivo(biblioteca.ruta(conjunto) / "demostraciones.npz")
    sub = _anadir_a_mano(biblioteca, _datos(conjunto, huella), fuente="humano")
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"]
    assert elemento["motivo"].startswith(f"su conjunto {conjunto} no es válido: la ficha dice")
