"""Carga de datos y splits.

REGLAS QUE NO SE ROMPEN (y que el revisor del PR comprueba):
  1. El split se hace ANTES de cualquier transformacion ajustada a los datos.
  2. La normalizacion usa estadisticas de ENTRENAMIENTO, nunca del conjunto
     completo. Lo contrario es una fuga: no da error e infla tu metrica.
  3. El data augmentation se aplica a train, jamas a validacion ni a test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.tipos import LoteTransiciones, Trayectoria, es_id_episodio


class DatasetPlaceholder(Dataset):
    """Sustituye esto por el dataset de tu paper.

    Se deja aqui un dataset sintetico para que `python run.py --smoke` funcione
    desde el minuto cero, antes de que hayas escrito nada. Es lo que permite que
    tengas el vertical slice en verde el primer dia.
    """

    def __init__(self, n: int = 512, dim: int = 32, clases: int = 10):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, clases, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def construir_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Devuelve (train_loader, val_loader) segun la configuracion."""
    dcfg = cfg["datos"]

    train_ds: Dataset = DatasetPlaceholder()
    val_ds: Dataset = DatasetPlaceholder(n=128)

    # Regla 5/20: el subset se aplica DESPUES del split, para no cambiar la
    # distribucion de validacion.
    if dcfg.get("subset"):
        n = int(dcfg["subset"])
        train_ds = Subset(train_ds, range(min(n, len(train_ds))))

    comun = dict(batch_size=dcfg["batch_size"],
                 num_workers=dcfg.get("num_workers", 0))
    return (DataLoader(train_ds, shuffle=True, **comun),
            DataLoader(val_ds, shuffle=False, **comun))


def submuestrear_pares(
    trayectorias: Sequence[Trayectoria], frecuencia: int, generador: np.random.Generator,
) -> LoteTransiciones:
    """Convierte partidas en pares (estado, acción) tomando uno de cada `frecuencia` pasos.

    Para cada partida se sortea un paso inicial en [0, min(frecuencia, T)) y se
    toman los pasos inicio, inicio + frecuencia, ... hasta el final, como hace el
    código original de GAIL. Limitar el inicio a T garantiza que toda partida
    aporte al menos un par. Con frecuencia 1 se toman todos los pasos. Los pares
    se devuelven en el orden de las partidas y, dentro de cada una, de sus pasos.

    Args:
        trayectorias: Secuencia no vacía de partidas ya validadas.
        frecuencia: Entero positivo; va en configs/.
        generador: Generador de src.utils.seed.crear_generador.

    Returns:
        Un LoteTransiciones con los estados, acciones, IDs de partida y pasos
        elegidos. Los arrays conservan el dtype de las partidas.

    Raises:
        ValueError: Frecuencia no válida, colección vacía o elementos que no son
            Trayectoria, id_episodio fuera del rango de int64 o generador no válido.
    """
    if (isinstance(frecuencia, (bool, np.bool_))
            or not isinstance(frecuencia, (int, np.integer)) or frecuencia < 1):
        raise ValueError(f"frecuencia debe ser un entero positivo, no {frecuencia!r}")
    if not isinstance(generador, np.random.Generator):
        raise ValueError("generador debe ser un numpy.random.Generator de crear_generador")
    if (isinstance(trayectorias, (str, bytes)) or not isinstance(trayectorias, Sequence)
            or not trayectorias or not all(isinstance(t, Trayectoria) for t in trayectorias)):
        raise ValueError("Se requiere una secuencia no vacía de objetos Trayectoria")
    if not all(es_id_episodio(t.id_episodio) for t in trayectorias):
        raise ValueError("Los id_episodio deben ser enteros en el rango de int64")

    estados, acciones, ids, pasos = [], [], [], []
    for trayectoria in trayectorias:
        inicio = int(generador.integers(0, min(int(frecuencia), trayectoria.numero_pasos)))
        elegidos = np.arange(inicio, trayectoria.numero_pasos, int(frecuencia))
        estados.append(trayectoria.estados[elegidos])
        acciones.append(trayectoria.acciones[elegidos])
        ids.append(np.full(len(elegidos), int(trayectoria.id_episodio), dtype=np.int64))
        pasos.append(elegidos.astype(np.int64))
    return LoteTransiciones(
        estados=np.concatenate(estados), acciones=np.concatenate(acciones),
        ids_episodio=np.concatenate(ids), pasos=np.concatenate(pasos),
    )

