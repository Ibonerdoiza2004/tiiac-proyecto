"""Selecciones de demostraciones: tramos de partidas, subconjuntos y combinaciones.

Lo que el dashboard hace con las demostraciones antes de entrenar: quedarse con un
tramo de una partida (al guardar lo jugado o al crear un subconjunto), juntar varios
conjuntos en uno y repartir sus partidas entre train y validación. Siempre se acaba
en partidas completas (Trayectoria) y un NPZ con sus splits, como en la E1.

Un tramo es ``[inicio, fin)`` en los pasos de la partida, desde 0 (comprobar_tramo).
"""

from __future__ import annotations

import numpy as np

from src.data.tipos import Trayectoria, comprobar_tramo


def recortar_partida(trayectoria: Trayectoria, tramo: object) -> Trayectoria:
    """Devuelve el tramo ``[inicio, fin)`` de una partida como partida propia.

    Conserva el id_episodio y copia los arrays (cambiar el resultado no cambia la
    partida). El formato exige que el último paso termine o se trunque: si el tramo
    llega al final de la partida, conserva sus marcas; si no, el paso de corte se
    marca truncado (cortado, no terminado por el entorno). BC y GAIL solo usan los
    pares, así que la marca no cambia lo que aprenden.

    Args:
        trayectoria: Partida completa.
        tramo: ``[inicio, fin)`` con ``0 <= inicio < fin <= pasos``.

    Returns:
        La partida recortada, con ``fin - inicio`` pasos.

    Raises:
        ValueError: Si no es una Trayectoria o el tramo no vale o no cabe.
    """
    if not isinstance(trayectoria, Trayectoria):
        raise ValueError("Se esperaba un objeto Trayectoria")
    inicio, fin = comprobar_tramo(tramo, trayectoria.numero_pasos)
    corte = slice(inicio, fin)
    terminados = np.array(trayectoria.terminados[corte], copy=True)
    truncados = np.array(trayectoria.truncados[corte], copy=True)
    if fin < trayectoria.numero_pasos:
        # Antes del final de una partida válida ningún paso termina: basta con truncarlo.
        truncados[-1] = True
    return Trayectoria(
        id_episodio=trayectoria.id_episodio,
        estados=np.array(trayectoria.estados[corte], copy=True),
        acciones=np.array(trayectoria.acciones[corte], copy=True),
        siguientes_estados=np.array(trayectoria.siguientes_estados[corte], copy=True),
        recompensas_originales=np.array(trayectoria.recompensas_originales[corte], copy=True),
        terminados=terminados,
        truncados=truncados,
    )
