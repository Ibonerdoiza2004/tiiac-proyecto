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
from gymnasium import spaces
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.splits import SplitsDemostraciones, validar_splits
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



def tensores_de_pares(estados: object, acciones: object,
                      espacio_accion: spaces.Space) -> tuple[torch.Tensor, torch.Tensor]:
    """Comprueba pares (estado, acción) frente al espacio y los pasa a tensores.

    Los estados salen float32 [N, d_observacion]. Con Discrete(n, start) las
    acciones salen int64 [N] en [0, n), ya sin el start, como esperan
    PoliticaBC y Discriminador; con Box salen float32 [N, d_accion]. Los
    arrays se copian y no se normalizan.

    Args:
        estados: Array de números finitos [N, d] con N > 0.
        acciones: Enteros [N] del espacio Discrete o reales finitos [N, d_accion]
            dentro de los límites del Box.
        espacio_accion: Discrete o Box de una dimensión del entorno.

    Returns:
        (estados, acciones) como tensores.

    Raises:
        ValueError: Pares vacíos o incoherentes, acciones fuera del espacio o
            espacio no compatible.
    """
    estados = np.asarray(estados)
    acciones = np.asarray(acciones)
    n = len(estados)
    if estados.ndim != 2 or n == 0 or estados.dtype.kind not in "iuf":
        raise ValueError(f"Los estados deben ser números [N, d] con N > 0, "
                         f"no {estados.dtype} {estados.shape}")
    if not np.all(np.isfinite(estados)):
        raise ValueError("Los estados deben ser finitos")

    if isinstance(espacio_accion, spaces.Discrete):
        if (acciones.shape != (n,) or not np.issubdtype(acciones.dtype, np.integer)):
            raise ValueError(f"Con acciones Discrete se esperaban enteros [{n}], "
                             f"no {acciones.dtype} {acciones.shape}")
        indices = acciones.astype(np.int64) - int(espacio_accion.start)
        if indices.min() < 0 or indices.max() >= int(espacio_accion.n):
            raise ValueError("Hay acciones fuera del espacio Discrete")
        tensor_acciones = torch.tensor(indices, dtype=torch.int64)
    elif isinstance(espacio_accion, spaces.Box) and len(espacio_accion.shape) == 1:
        forma = (n, int(espacio_accion.shape[0]))
        if acciones.shape != forma or not np.issubdtype(acciones.dtype, np.floating):
            raise ValueError(f"Con acciones Box se esperaban reales {list(forma)}, "
                             f"no {acciones.dtype} {acciones.shape}")
        if not np.all(np.isfinite(acciones)):
            raise ValueError("Las acciones deben ser finitas")
        if np.any(acciones < espacio_accion.low) or np.any(acciones > espacio_accion.high):
            raise ValueError("Hay acciones fuera de los límites del espacio Box")
        tensor_acciones = torch.tensor(acciones, dtype=torch.float32)
    else:
        raise ValueError(
            "Espacio de acción no compatible: se admite Discrete o Box de una dimensión")
    return torch.tensor(estados, dtype=torch.float32), tensor_acciones


class DatasetTransiciones(Dataset):
    """Pares (estado, acción) de un LoteTransiciones como tensores para BC.

    Los estados son float32 [N, d_observacion]. Con Discrete(n, start) las
    acciones son int64 [N] en [0, n), ya sin el start, como espera PoliticaBC;
    con Box son float32 [N, d_accion]. ids_episodio conserva de qué partida
    sale cada par. Los arrays se copian y no se normalizan (tensores_de_pares).
    """

    def __init__(self, lote: LoteTransiciones, espacio_accion: spaces.Space) -> None:
        """Comprueba el lote frente al espacio de acción y lo convierte a tensores.

        Args:
            lote: Pares no vacíos, por ejemplo de submuestrear_pares.
            espacio_accion: Discrete o Box de una dimensión del entorno.

        Raises:
            ValueError: Lote vacío o incoherente, acciones fuera del espacio o
                espacio no compatible.
        """
        if not isinstance(lote, LoteTransiciones):
            raise ValueError("Se esperaba un LoteTransiciones")
        self.estados, self.acciones = tensores_de_pares(lote.estados, lote.acciones,
                                                        espacio_accion)
        if np.shape(lote.ids_episodio) != (len(self.estados),):
            raise ValueError(f"ids_episodio debe tener forma [{len(self.estados)}]")
        self.ids_episodio = np.array(lote.ids_episodio, dtype=np.int64)

    def __len__(self) -> int:
        """Devuelve el número de pares."""
        return len(self.estados)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve el par (estado, acción) en la posición i."""
        return self.estados[i], self.acciones[i]


def construir_loaders_bc(
    trayectorias: Sequence[Trayectoria], splits: SplitsDemostraciones, cantidad: int,
    espacio_accion: spaces.Space, cfg: dict, generador: np.random.Generator,
) -> tuple[DataLoader, DataLoader]:
    """Devuelve (train_loader, val_loader) de BC para un subconjunto de partidas.

    Train usa los pares submuestreados de las partidas del subconjunto cantidad
    y validación los de las partidas de validación, con la misma frecuencia, de
    modo que ninguna partida aporta pares a ambos. Con el mismo generador se
    sortean primero los pares de validación y después los de train. Así, con la
    misma semilla, validación tiene los mismos pares para cualquier cantidad, y
    las partidas que comparten dos subconjuntos aportan los mismos pares de
    train en ambos. Train se baraja en cada época y validación no.

    Args:
        trayectorias: Todas las partidas del dataset para el que se crearon los splits.
        splits: Splits de esas partidas.
        cantidad: Número de partidas de train que se usan, por ejemplo 1, 4, 7 o 10.
            Tiene que haber un subconjunto de ese tamaño: splits.subconjuntos
            asocia cada tamaño con los IDs de sus partidas.
        espacio_accion: Espacio de acción del entorno.
        cfg: Configuración resuelta con bc.frecuencia_pares y bc.batch_size.
        generador: Generador de src.utils.seed.crear_generador.

    Raises:
        ValueError: Splits que no corresponden a las partidas, cantidad sin
            subconjunto, configuración incompleta o datos no válidos.
    """
    bcfg = cfg.get("bc") if isinstance(cfg, dict) else None
    if not isinstance(bcfg, dict) or not {"frecuencia_pares", "batch_size"} <= bcfg.keys():
        raise ValueError("La configuración necesita bc.frecuencia_pares y bc.batch_size")
    batch_size = bcfg["batch_size"]
    if (isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1):
        raise ValueError(f"bc.batch_size debe ser un entero positivo, no {batch_size!r}")
    if (isinstance(trayectorias, (str, bytes)) or not isinstance(trayectorias, Sequence)
            or not all(isinstance(t, Trayectoria) for t in trayectorias)):
        raise ValueError("Se requiere una secuencia de objetos Trayectoria")
    validar_splits(splits, [t.id_episodio for t in trayectorias])
    if (isinstance(cantidad, (bool, np.bool_)) or not isinstance(cantidad, (int, np.integer))
            or cantidad not in splits.subconjuntos):
        raise ValueError(f"No hay subconjunto de {cantidad!r} partidas; "
                         f"disponibles: {sorted(splits.subconjuntos)}")

    por_id = {int(t.id_episodio): t for t in trayectorias}
    frecuencia = bcfg["frecuencia_pares"]
    validacion = submuestrear_pares([por_id[i] for i in splits.validacion], frecuencia, generador)
    train = submuestrear_pares([por_id[i] for i in splits.subconjuntos[cantidad]],
                               frecuencia, generador)
    return (DataLoader(DatasetTransiciones(train, espacio_accion),
                       batch_size=int(batch_size), shuffle=True),
            DataLoader(DatasetTransiciones(validacion, espacio_accion),
                       batch_size=int(batch_size), shuffle=False))
