"""Construccion del modelo.

Aqui va tu replicacion. El objetivo de E1 es que el forward produzca las
FORMAS que declara el paper, aunque los pesos sean aleatorios.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical, Distribution, Independent, Normal

ACTIVACIONES = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class ModeloPlaceholder(nn.Module):
    """Sustituye esto por la arquitectura del paper."""

    def __init__(self, dim_entrada: int = 32, clases: int = 10):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, 64),
            nn.ReLU(),
            nn.Linear(64, clases),
        )

    def forward(self, x):
        return self.red(x)


def _es_entero_positivo(valor: object) -> bool:
    """Indica si valor es un entero no booleano mayor que cero."""
    return (isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))
            and valor > 0)


def _es_real_finito(valor: object) -> bool:
    """Indica si valor es un número real finito no booleano."""
    return (isinstance(valor, (int, float, np.integer, np.floating))
            and not isinstance(valor, (bool, np.bool_)) and bool(np.isfinite(valor)))


class PoliticaBC(nn.Module):
    """Política de Behavioral Cloning para acciones discretas o continuas.

    Un perceptrón multicapa recibe estados [B, dim_observacion] y devuelve
    [B, dim_accion]: los logits de una distribución categórica con acciones
    Discrete, o la media de una gaussiana diagonal con acciones Box. En el caso
    continuo la desviación típica no depende del estado: es el parámetro
    aprendido log_std, de forma [dim_accion], como en las políticas gaussianas
    de TRPO. BC maximiza la log-verosimilitud de las acciones del experto, que
    con acciones discretas equivale a la entropía cruzada.

    Antes de la red, cada componente del estado se normaliza con la media y la
    escala guardadas en los buffers media y escala, que viajan con el
    state_dict y por tanto con los checkpoints. Valen 0 y 1 (identidad) hasta
    que se llama a ajustar_normalizacion con estados de train.

    Con acciones discretas el índice i corresponde a la acción start + i del
    espacio Discrete; la traducción la hace quien prepara los datos o elige la
    acción. Los pesos se inicializan como en el código original de GAIL
    (policyopt/nn.py), ya que el paper no lo especifica: Glorot uniforme en las
    capas ocultas, la capa de salida a cero y todos los sesgos a cero. Así la
    política inicial es uniforme sobre las acciones discretas o tiene media cero.
    """

    def __init__(self, dim_observacion: int, dim_accion: int, continua: bool,
                 capas_ocultas: Sequence[int], activacion: str,
                 log_std_inicial: float | None = None) -> None:
        """Crea la red.

        Args:
            dim_observacion: Entero positivo, tamaño del estado.
            dim_accion: Entero positivo: n del espacio Discrete o tamaño del Box.
            continua: True para acciones Box y False para Discrete.
            capas_ocultas: Tamaños enteros positivos de cada capa oculta; vacía
                da una política lineal. Va en configs/.
            activacion: Clave de ACTIVACIONES. Va en configs/.
            log_std_inicial: Valor inicial finito de log_std; obligatorio con
                acciones continuas y None con discretas. Va en configs/.

        Raises:
            ValueError: Si algún argumento no es válido.
        """
        super().__init__()
        if not _es_entero_positivo(dim_observacion):
            raise ValueError(
                f"dim_observacion debe ser un entero positivo, no {dim_observacion!r}")
        if not _es_entero_positivo(dim_accion):
            raise ValueError(f"dim_accion debe ser un entero positivo, no {dim_accion!r}")
        if not isinstance(continua, (bool, np.bool_)):
            raise ValueError(f"continua debe ser un booleano, no {continua!r}")
        if (isinstance(capas_ocultas, (str, bytes)) or not isinstance(capas_ocultas, Sequence)
                or not all(_es_entero_positivo(c) for c in capas_ocultas)):
            raise ValueError(
                f"capas_ocultas debe ser una lista de enteros positivos, no {capas_ocultas!r}")
        if not isinstance(activacion, str) or activacion not in ACTIVACIONES:
            raise ValueError(
                f"activacion desconocida: {activacion!r}; opciones: {sorted(ACTIVACIONES)}")
        if continua and not _es_real_finito(log_std_inicial):
            raise ValueError("Con acciones continuas log_std_inicial debe ser un real finito, "
                             f"no {log_std_inicial!r}")
        if not continua and log_std_inicial is not None:
            raise ValueError("Con acciones discretas log_std_inicial debe ser None")

        self.dim_observacion = int(dim_observacion)
        self.dim_accion = int(dim_accion)
        self.continua = bool(continua)
        capas: list[nn.Module] = []
        entrada = self.dim_observacion
        for tamano in capas_ocultas:
            capas += [nn.Linear(entrada, int(tamano)), ACTIVACIONES[activacion]()]
            entrada = int(tamano)
        capas.append(nn.Linear(entrada, self.dim_accion))
        self.red = nn.Sequential(*capas)
        lineales = [capa for capa in self.red if isinstance(capa, nn.Linear)]
        for capa in lineales[:-1]:
            nn.init.xavier_uniform_(capa.weight)
        nn.init.zeros_(lineales[-1].weight)
        for capa in lineales:
            nn.init.zeros_(capa.bias)
        self.log_std = (nn.Parameter(torch.full((self.dim_accion,), float(log_std_inicial)))
                        if self.continua else None)
        self.register_buffer("media", torch.zeros(self.dim_observacion))
        self.register_buffer("escala", torch.ones(self.dim_observacion))

    def ajustar_normalizacion(self, estados: torch.Tensor, epsilon: float) -> None:
        """Fija la normalización de observaciones a partir de estados de train.

        Como el Standardizer del código original de GAIL, resta la media y divide
        por la desviación típica poblacional de cada componente, calculadas en
        float64. A diferencia del original, que divide por desviación + epsilon,
        una componente con desviación <= epsilon solo se centra (escala 1): así
        no se multiplican por 1/epsilon las variaciones que aparezcan fuera de
        train. Con un único estado todas las componentes solo se centran. Las
        estadísticas deben salir solo de train para no filtrar validación ni test.

        Args:
            estados: Tensor real finito [N, dim_observacion] con N >= 1.
            epsilon: Real positivo finito. Va en configs/.

        Raises:
            ValueError: Si estados o epsilon no son válidos.
        """
        if (not isinstance(estados, torch.Tensor) or estados.ndim != 2
                or estados.shape[1] != self.dim_observacion or len(estados) == 0
                or not estados.is_floating_point()):
            raise ValueError(f"Se esperaban estados reales [N, {self.dim_observacion}] con N > 0")
        if not torch.isfinite(estados).all():
            raise ValueError("Los estados para normalizar deben ser finitos")
        if not _es_real_finito(epsilon) or epsilon <= 0:
            raise ValueError(f"epsilon debe ser un real positivo finito, no {epsilon!r}")
        datos = estados.detach().to(torch.float64)
        media = datos.mean(dim=0)
        desviacion = datos.std(dim=0, unbiased=False)
        escala = torch.where(desviacion > float(epsilon), desviacion, torch.ones_like(desviacion))
        self.media.copy_(media)
        self.escala.copy_(escala)

    def forward(self, estados: torch.Tensor) -> torch.Tensor:
        """Devuelve logits o medias [B, dim_accion] de estados [B, dim_observacion].

        Raises:
            ValueError: Si estados no tiene forma [B, dim_observacion].
        """
        if estados.ndim != 2 or estados.shape[1] != self.dim_observacion:
            raise ValueError(f"Se esperaban estados [B, {self.dim_observacion}], "
                             f"no {tuple(estados.shape)}")
        return self.red((estados - self.media) / self.escala)

    def distribucion(self, estados: torch.Tensor) -> Distribution:
        """Devuelve la distribución de las acciones para estados [B, dim_observacion].

        Es Categorical con batch_shape [B] para acciones discretas e
        Independent(Normal) con batch_shape [B] y event_shape [dim_accion] para
        continuas.
        """
        salida = self(estados)
        if not self.continua:
            return Categorical(logits=salida, validate_args=len(salida) > 0)
        return Independent(self._normal(salida), 1, validate_args=len(salida) > 0)

    def _normal(self, medias: torch.Tensor) -> Normal:
        """Gaussiana por componente con media medias [B, dim_accion] y desviación exp(log_std)."""
        # La validación interna de torch.distributions falla con un batch vacío.
        return Normal(medias, self.log_std.exp().expand_as(medias), validate_args=len(medias) > 0)

    def log_verosimilitud(self, estados: torch.Tensor, acciones: torch.Tensor) -> torch.Tensor:
        """Devuelve el log de la probabilidad [B] de cada acción en su estado.

        Args:
            estados: Tensor real [B, dim_observacion].
            acciones: Con acciones discretas, enteros [B] en [0, dim_accion), ya
                sin el start del espacio. Con continuas, reales [B, dim_accion].

        Raises:
            ValueError: Si las acciones no tienen la forma, el tipo o el rango esperados.
        """
        if self.continua:
            if acciones.shape != (len(estados), self.dim_accion) or not acciones.is_floating_point():
                raise ValueError(f"Se esperaban acciones reales [{len(estados)}, "
                                 f"{self.dim_accion}], no {acciones.dtype} {tuple(acciones.shape)}")
        else:
            if (acciones.shape != (len(estados),) or acciones.is_floating_point()
                    or acciones.is_complex() or acciones.dtype == torch.bool):
                raise ValueError(f"Se esperaban acciones enteras [{len(estados)}], "
                                 f"no {acciones.dtype} {tuple(acciones.shape)}")
            if len(acciones) and (acciones.min() < 0 or acciones.max() >= self.dim_accion):
                raise ValueError(f"Las acciones discretas deben estar en [0, {self.dim_accion})")
        if self.continua:
            # Equivale a distribucion(...).log_prob, que con un batch vacío falla en
            # Independent: se suman las componentes de la gaussiana diagonal.
            return self._normal(self(estados)).log_prob(acciones).sum(dim=1)
        return self.distribucion(estados).log_prob(acciones)


def construir_politica_bc(cfg: dict, espacio_observacion: spaces.Space,
                          espacio_accion: spaces.Space) -> PoliticaBC:
    """Construye la política de BC a partir de la configuración y los espacios.

    Args:
        cfg: Configuración resuelta con la sección bc: capas_ocultas y
            activacion, y además log_std_inicial con acciones continuas.
        espacio_observacion: Box de una dimensión del entorno.
        espacio_accion: Discrete o Box de una dimensión del entorno.

    Returns:
        Una PoliticaBC con dim_observacion y dim_accion tomados de los espacios.

    Raises:
        ValueError: Si falta la sección bc o alguna clave necesaria, o si los
            espacios no son compatibles.
    """
    bcfg = cfg.get("bc") if isinstance(cfg, dict) else None
    if not isinstance(bcfg, dict) or not {"capas_ocultas", "activacion"} <= bcfg.keys():
        raise ValueError("La configuración necesita bc.capas_ocultas y bc.activacion")
    if not isinstance(espacio_observacion, spaces.Box) or len(espacio_observacion.shape) != 1:
        raise ValueError("Espacio de observación no compatible: se admite Box de una dimensión")
    dim_observacion = int(espacio_observacion.shape[0])
    if isinstance(espacio_accion, spaces.Discrete):
        return PoliticaBC(dim_observacion, int(espacio_accion.n), False,
                          bcfg["capas_ocultas"], bcfg["activacion"])
    if isinstance(espacio_accion, spaces.Box) and len(espacio_accion.shape) == 1:
        if "log_std_inicial" not in bcfg:
            raise ValueError("Con acciones continuas la configuración necesita bc.log_std_inicial")
        return PoliticaBC(dim_observacion, int(espacio_accion.shape[0]), True,
                          bcfg["capas_ocultas"], bcfg["activacion"], bcfg["log_std_inicial"])
    raise ValueError(
        "Espacio de acción no compatible: se admite Discrete o Box de una dimensión")


def construir_modelo(cfg: dict) -> nn.Module:
    nombre = cfg["modelo"]["nombre"]
    if nombre == "mi_modelo":
        return ModeloPlaceholder()
    raise ValueError(f"modelo desconocido: {nombre}")
