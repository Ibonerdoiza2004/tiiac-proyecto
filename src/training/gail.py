"""Recompensa de GAIL: discriminador, su entrenamiento y el envoltorio del entorno.

Sigue el código original de GAIL (openai/imitation, TransitionClassifier):

- Durante un bloque de recogida el discriminador está fijo y en modo
  evaluación; cada paso recibe la recompensa que calcula a partir del par
  (estado, acción) en lugar de la del entorno (EntornoRecompensaGAIL).
- Después del bloque se actualiza con los pares del agente frente a otros
  tantos pares expertos muestreados con reemplazo (muestrear_expertos y
  actualizar_discriminador).

- Después de actualizarlo, reajustar_valor_tras_discriminador pone al día la red
  de valor de TRPO con las recompensas del discriminador nuevo, como el original.

iteracion_gail encadena una iteración completa: un bloque de TRPO con la
recompensa del discriminador, su actualización y el reajuste de la red de valor.
entrenar_gail monta las piezas de una repetición con un subconjunto de partidas
y repite la iteración, registrando un retorno de validación cada cierto tiempo.
guardar_gail y cargar_gail guardan el resultado como el experto, con el
discriminador y la procedencia de los datos (procedencia_datos, en barrido.py).
entrenar_y_guardar_gail es un entrenamiento del barrido completo (run de W&B,
progreso, entrenamiento y guardado): la tarea que scripts/entrenar_gail.py
reparte con ejecutar_barrido (barrido.py).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import TRPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from src.data.entornos import crear_entorno_vectorizado
from src.data.loaders import pares_de_subconjunto, tensores_de_pares
from src.data.splits import SplitsDemostraciones, huella_archivo
from src.data.tipos import LoteTransiciones, Trayectoria
from src.models.factory import Discriminador, comprobar_arquitectura, construir_discriminador
from src.training.barrido import MetodoBarrido, comprobar_config_comun
from src.training.trpo import (SECCION_TRPO, config_trpo, crear_trpo,
                               estadisticas_normalizacion, evaluar_retorno, iteracion_trpo)
from src.utils.archivos import comprobar_directorio_libre, escribir_carpeta_nueva
from src.utils.procedencia import a_json, estado_git, versiones_y_hardware
from src.utils.seed import crear_generador, fijar_seed, semillas_repeticion
from src.utils.tracking import (RegistroProgreso, abrir_run_barrido, con_seccion,
                                nombre_artefacto, registrar, registrar_resumen,
                                subir_artefacto, terminar_run, usar_artefacto)
from src.utils.validacion import es_entero_positivo, es_real_finito

_ARCHIVO_MODELO = "modelo.zip"
_ARCHIVO_DISCRIMINADOR = "discriminador.pt"
_ARCHIVO_METADATOS = "gail.json"

# −log(1 − D) da recompensas >= 0: favorece alargar las partidas, para entornos
# que terminan al fallar (CartPole). log D da recompensas <= 0: favorece
# acabarlas, para entornos que terminan al tener éxito (MountainCar, Acrobot).
# En el original, la segunda es favor_zero_expert_reward (cuts_off_on_success).
MODOS_RECOMPENSA = ("menos_log_uno_menos_d", "log_d")


def _config_discriminador(cfg: dict) -> dict:
    """Devuelve gail.discriminador comprobado.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    gcfg = cfg.get("gail") if isinstance(cfg, dict) else None
    dcfg = gcfg.get("discriminador") if isinstance(gcfg, dict) else None
    claves = {"capas_ocultas", "activacion", "optimizador", "learning_rate", "pasos_adam",
              "peso_entropia", "normalizar_entrada", "epsilon_normalizacion"}
    if not isinstance(dcfg, dict) or not claves <= dcfg.keys():
        raise ValueError("La configuración necesita gail.discriminador con "
                         + ", ".join(sorted(claves)))
    # La misma comprobación que al crear la red: así comprobar_config_barrido la
    # rechaza antes de lanzar el barrido y no en cada entrenamiento.
    try:
        comprobar_arquitectura(dcfg["capas_ocultas"], dcfg["activacion"])
    except ValueError as error:
        raise ValueError(f"gail.discriminador no es válido: {error}") from error

    def real(clave: str) -> float:
        """Devuelve dcfg[clave] si es un real finito no booleano."""
        valor = dcfg[clave]
        if not es_real_finito(valor):
            raise ValueError(f"gail.discriminador.{clave} debe ser un real finito, no {valor!r}")
        return float(valor)

    if dcfg["optimizador"] != "adam":
        raise ValueError(f"gail.discriminador.optimizador desconocido: {dcfg['optimizador']!r}; "
                         "opciones: ['adam']")
    if real("learning_rate") <= 0:
        raise ValueError("gail.discriminador.learning_rate debe ser positivo")
    pasos = dcfg["pasos_adam"]
    if not es_entero_positivo(pasos):
        raise ValueError(
            f"gail.discriminador.pasos_adam debe ser un entero positivo, no {pasos!r}")
    if real("peso_entropia") < 0:
        raise ValueError("gail.discriminador.peso_entropia no puede ser negativo")
    if not isinstance(dcfg["normalizar_entrada"], bool):
        raise ValueError("gail.discriminador.normalizar_entrada debe ser un booleano")
    if real("epsilon_normalizacion") <= 0:
        raise ValueError("gail.discriminador.epsilon_normalizacion debe ser positivo")
    return dcfg


def _comprobar_lote(lote: object, nombre: str) -> LoteTransiciones:
    """Comprueba que lote es un LoteTransiciones con arrays de la misma longitud."""
    if not isinstance(lote, LoteTransiciones):
        raise ValueError(f"{nombre} debe ser un LoteTransiciones")
    longitudes = {len(np.asarray(lote.estados)), len(np.asarray(lote.acciones)),
                  len(np.asarray(lote.ids_episodio)), len(np.asarray(lote.pasos))}
    if len(longitudes) != 1:
        raise ValueError(f"{nombre} tiene arrays de longitudes distintas")
    return lote


def preparar_discriminador(cfg: dict, espacio_observacion: spaces.Space,
                           espacio_accion: spaces.Space, pares_expertos: LoteTransiciones,
                           ) -> tuple[Discriminador, torch.optim.Optimizer]:
    """Crea un discriminador nuevo y su optimizador para una ejecución de GAIL.

    Si gail.discriminador.normalizar_entrada es true, la normalización de la
    entrada se fija con los pares expertos de train y no cambia después. Es una
    elección propia: el original empieza con los datos expertos pero la va
    actualizando en cada iteración también con pares del agente.

    Args:
        cfg: Configuración resuelta con la sección gail.
        espacio_observacion: Box de una dimensión del entorno.
        espacio_accion: Discrete o Box de una dimensión del entorno.
        pares_expertos: Pares expertos del subconjunto de train.

    Returns:
        (discriminador, optimizador Adam con gail.discriminador.learning_rate).

    Raises:
        ValueError: Si la configuración, los espacios o los pares no son válidos.
    """
    dcfg = _config_discriminador(cfg)
    discriminador = construir_discriminador(cfg, espacio_observacion, espacio_accion)
    if dcfg["normalizar_entrada"]:
        _comprobar_lote(pares_expertos, "pares_expertos")
        estados, acciones = tensores_de_pares(pares_expertos.estados, pares_expertos.acciones,
                                              espacio_accion)
        discriminador.ajustar_normalizacion(estados, acciones, dcfg["epsilon_normalizacion"])
    optimizador = torch.optim.Adam(discriminador.parameters(), lr=float(dcfg["learning_rate"]))
    return discriminador, optimizador


def calcular_recompensas_gail(discriminador: Discriminador, estados: np.ndarray,
                              acciones: np.ndarray, espacio_accion: spaces.Space,
                              modo: str) -> np.ndarray:
    """Calcula la recompensa de GAIL de cada par, en modo evaluación y sin gradientes.

    Con logit l y D = sigmoid(l), menos_log_uno_menos_d da −log(1 − D) =
    softplus(l) y log_d da log D = logsigmoid(l); así se calculan sin pasar
    por D, que con logits grandes se redondea a 0 o 1 y daría infinitos.

    Args:
        discriminador: Se deja en modo evaluación.
        estados: Array [N, d_observacion].
        acciones: Acciones del entorno: enteros [N] del Discrete (con su start)
            o reales [N, d_accion] del Box.
        espacio_accion: Espacio de acción del entorno.
        modo: Uno de MODOS_RECOMPENSA.

    Returns:
        Recompensas float32 [N].

    Raises:
        ValueError: Si el modo o los pares no son válidos.
    """
    if modo not in MODOS_RECOMPENSA:
        raise ValueError(f"Modo de recompensa desconocido: {modo!r}; opciones: "
                         f"{list(MODOS_RECOMPENSA)}")
    tensor_estados, tensor_acciones = tensores_de_pares(estados, acciones, espacio_accion)
    discriminador.eval()
    with torch.no_grad():
        logits = discriminador(tensor_estados, tensor_acciones)
        recompensas = (F.softplus(logits) if modo == "menos_log_uno_menos_d"
                       else F.logsigmoid(logits))
    return recompensas.numpy().astype(np.float32)


def muestrear_expertos(pares: LoteTransiciones, cantidad: int,
                       generador: np.random.Generator) -> LoteTransiciones:
    """Elige cantidad pares expertos al azar, con reemplazo, como el original.

    Args:
        pares: Pares expertos no vacíos, por ejemplo de submuestrear_pares.
        cantidad: Entero positivo; normalmente, el número de pares del agente.
        generador: Generador de crear_generador; es lo único aleatorio que se usa.

    Returns:
        Un LoteTransiciones nuevo con los pares elegidos, sus ids y sus pasos.

    Raises:
        ValueError: Si algún argumento no es válido.
    """
    _comprobar_lote(pares, "pares")
    total = len(np.asarray(pares.estados))
    if total == 0:
        raise ValueError("No hay pares expertos que muestrear")
    if not es_entero_positivo(cantidad):
        raise ValueError(f"cantidad debe ser un entero positivo, no {cantidad!r}")
    if not isinstance(generador, np.random.Generator):
        raise ValueError("generador debe ser un numpy.random.Generator de crear_generador")
    indices = generador.integers(0, total, size=int(cantidad))
    return LoteTransiciones(*(np.asarray(array)[indices] for array in
                              (pares.estados, pares.acciones, pares.ids_episodio, pares.pasos)))


def _entropia_bernoulli(logits: torch.Tensor) -> torch.Tensor:
    """Entropía de Bernoulli(sigmoid(logits)) por elemento: softplus(l) − l·sigmoid(l)."""
    return F.softplus(logits) - logits * torch.sigmoid(logits)


def _perdida_grupo(logits: torch.Tensor, etiqueta: float, peso_entropia: float) -> torch.Tensor:
    """Media de la entropía cruzada menos peso_entropia por la entropía, de un grupo."""
    objetivo = torch.full_like(logits, etiqueta)
    cruzada = F.binary_cross_entropy_with_logits(logits, objetivo, reduction="none")
    return (cruzada - peso_entropia * _entropia_bernoulli(logits)).mean()


def actualizar_discriminador(discriminador: Discriminador, optimizador: torch.optim.Optimizer,
                             lote_agente: LoteTransiciones, lote_experto: LoteTransiciones,
                             espacio_accion: spaces.Space, cfg: dict) -> dict[str, float]:
    """Entrena el discriminador para separar los pares expertos (1) de los del agente (0).

    Como el original, da gail.discriminador.pasos_adam pasos del optimizador
    sobre la regresión logística regularizada con la entropía:
    media(BCE − peso·H) del agente + media(BCE − peso·H) del experto. Cada
    grupo pesa lo mismo aunque tengan tamaños distintos (pesos 1/B por grupo).

    Args:
        discriminador: Se deja en modo entrenamiento.
        optimizador: Se conserva entre iteraciones; viene de preparar_discriminador.
        lote_agente: Pares del agente del último bloque (extraer_lote).
        lote_experto: Pares expertos, por ejemplo de muestrear_expertos.
        espacio_accion: Espacio de acción del entorno.
        cfg: Configuración resuelta con la sección gail.

    Returns:
        Medidas del último paso, antes de aplicarlo: perdida, entropia (media
        de las medias de cada grupo) y acierto_agente / acierto_experto
        (fracción de pares de cada grupo con el logit del lado correcto de 0).

    Raises:
        ValueError: Si la configuración o los pares no son válidos.
    """
    dcfg = _config_discriminador(cfg)
    _comprobar_lote(lote_agente, "lote_agente")
    _comprobar_lote(lote_experto, "lote_experto")
    estados_agente, acciones_agente = tensores_de_pares(lote_agente.estados,
                                                        lote_agente.acciones, espacio_accion)
    estados_experto, acciones_experto = tensores_de_pares(lote_experto.estados,
                                                          lote_experto.acciones, espacio_accion)
    peso = float(dcfg["peso_entropia"])
    discriminador.train()
    for _ in range(dcfg["pasos_adam"]):
        logits_agente = discriminador(estados_agente, acciones_agente)
        logits_experto = discriminador(estados_experto, acciones_experto)
        perdida = (_perdida_grupo(logits_agente, 0.0, peso)
                   + _perdida_grupo(logits_experto, 1.0, peso))
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
    with torch.no_grad():
        # Cada grupo pesa lo mismo, como en la pérdida.
        entropia = (_entropia_bernoulli(logits_agente).mean()
                    + _entropia_bernoulli(logits_experto).mean()) / 2
        return {
            "perdida": perdida.item(),
            "entropia": entropia.item(),
            "acierto_agente": (logits_agente < 0).float().mean().item(),
            "acierto_experto": (logits_experto > 0).float().mean().item(),
        }


class EntornoRecompensaGAIL(VecEnvWrapper):
    """VecEnv que da la recompensa del discriminador y guarda los pares jugados.

    Cada paso puntúa el par (estado previo, acción ejecutada) con
    calcular_recompensas_gail y devuelve esa recompensa en lugar de la del
    entorno, que queda en infos[i]["recompensa_original"]. Si los entornos de
    debajo llevan Monitor, el retorno de los episodios sigue siendo el real.

    Los pares se acumulan hasta extraer_lote, en orden de recogida (paso a paso
    y, dentro de cada paso, entorno a entorno), con un id por partida y el
    número de paso dentro de ella. Tras una partida terminada, el estado
    previo del paso siguiente es la primera observación de la partida nueva,
    que es la que devuelve el VecEnv con su reinicio automático.

    El discriminador no se entrena aquí: debe quedar fijo durante el bloque.
    """

    def __init__(self, venv: VecEnv, discriminador: Discriminador, modo: str) -> None:
        """Envuelve venv.

        Args:
            venv: VecEnv con observaciones Box de una dimensión y acciones
                Discrete con start 0 o Box de una dimensión, compatibles con el
                discriminador. sb3 no admite Discrete con otro start: sus
                políticas generan índices 0..n-1, que el envoltorio pasa tal cual.
            discriminador: Discriminador de construir_discriminador.
            modo: Uno de MODOS_RECOMPENSA (gail.recompensa del perfil).

        Raises:
            ValueError: Si los espacios, el discriminador o el modo no son compatibles.
        """
        if modo not in MODOS_RECOMPENSA:
            raise ValueError(f"Modo de recompensa desconocido: {modo!r}; opciones: "
                             f"{list(MODOS_RECOMPENSA)}")
        if not isinstance(discriminador, Discriminador):
            raise ValueError("discriminador debe ser un Discriminador")
        observacion, accion = venv.observation_space, venv.action_space
        if (not isinstance(observacion, spaces.Box) or observacion.shape
                != (discriminador.dim_observacion,)):
            raise ValueError("El espacio de observación no coincide con el discriminador")
        discreta = isinstance(accion, spaces.Discrete)
        dim_accion = (int(accion.n) if discreta else
                      accion.shape[0] if isinstance(accion, spaces.Box) and len(accion.shape) == 1
                      else None)
        if dim_accion != discriminador.dim_accion or discreta == discriminador.continua:
            raise ValueError("El espacio de acción no coincide con el discriminador")
        if discreta and int(accion.start) != 0:
            raise ValueError("Acciones Discrete con start distinto de 0: sb3 solo genera índices "
                             "0..n-1 y no admite estos espacios")
        super().__init__(venv)
        self.discriminador = discriminador
        self.modo = modo
        self._observaciones: np.ndarray | None = None
        self._acciones: np.ndarray | None = None
        self._ids = np.zeros(self.num_envs, dtype=np.int64)
        self._pasos = np.zeros(self.num_envs, dtype=np.int64)
        self._siguiente_id = 0
        self.iniciar_bloque()

    def _nuevos_ids(self, cuantos: int) -> np.ndarray:
        """Reserva cuantos ids de partida consecutivos."""
        ids = np.arange(self._siguiente_id, self._siguiente_id + cuantos, dtype=np.int64)
        self._siguiente_id += cuantos
        return ids

    def reset(self) -> VecEnvObs:
        """Reinicia todos los entornos; cada uno empieza una partida con id nuevo."""
        observaciones = self.venv.reset()
        self._observaciones = np.array(observaciones, copy=True)
        self._ids = self._nuevos_ids(self.num_envs)
        self._pasos[:] = 0
        return observaciones

    def step_async(self, actions: np.ndarray) -> None:
        """Guarda las acciones que se ejecutan y las pasa al VecEnv.

        Raises:
            RuntimeError: Si no se ha llamado antes a reset.
        """
        if self._observaciones is None:
            raise RuntimeError("Hay que llamar a reset antes del primer paso")
        # El entorno recibe la misma copia que se puntúa y se guarda: si quien
        # llama cambia su array antes de step_wait, no se desincronizan.
        self._acciones = np.array(actions, copy=True)
        self.venv.step_async(self._acciones)

    def step_wait(self) -> VecEnvStepReturn:
        """Da el paso y sustituye la recompensa por la del discriminador."""
        observaciones, recompensas_originales, terminados, infos = self.venv.step_wait()
        recompensas = calcular_recompensas_gail(self.discriminador, self._observaciones,
                                                self._acciones, self.action_space, self.modo)
        for info, original in zip(infos, recompensas_originales, strict=True):
            info["recompensa_original"] = float(original)
        self._estados_bloque.append(self._observaciones)
        self._recompensas_bloque.append(np.array(recompensas, copy=True))
        self._acciones_bloque.append(self._acciones)
        self._ids_bloque.append(self._ids.copy())
        self._pasos_bloque.append(self._pasos.copy())

        self._pasos += 1
        acabados = np.flatnonzero(terminados)
        self._ids[acabados] = self._nuevos_ids(len(acabados))
        self._pasos[acabados] = 0
        self._observaciones = np.array(observaciones, copy=True)
        return observaciones, recompensas, terminados, infos

    def iniciar_bloque(self) -> None:
        """Descarta los pares y las recompensas acumulados para empezar un bloque nuevo."""
        self._estados_bloque: list[np.ndarray] = []
        self._recompensas_bloque: list[np.ndarray] = []
        self._acciones_bloque: list[np.ndarray] = []
        self._ids_bloque: list[np.ndarray] = []
        self._pasos_bloque: list[np.ndarray] = []

    def extraer_bloque(self) -> tuple[LoteTransiciones, np.ndarray]:
        """Devuelve los pares y las recompensas acumulados desde iniciar_bloque y los descarta.

        Las recompensas son las que dio el discriminador en cada paso, con forma
        [pasos, entornos]: la misma que el rollout_buffer de sb3, que las guarda
        en el mismo orden. El par i del lote corresponde a la recompensa
        [i // entornos, i % entornos]. reajustar_valor_tras_discriminador las
        necesita para cambiarlas por las del discriminador actualizado.

        Raises:
            ValueError: Si no se ha dado ningún paso desde iniciar_bloque.
        """
        if not self._estados_bloque:
            raise ValueError("No hay pares: no se ha dado ningún paso en el bloque")
        lote = LoteTransiciones(np.concatenate(self._estados_bloque),
                                np.concatenate(self._acciones_bloque),
                                np.concatenate(self._ids_bloque),
                                np.concatenate(self._pasos_bloque))
        recompensas = np.stack(self._recompensas_bloque)
        self.iniciar_bloque()
        return lote, recompensas

    def extraer_lote(self) -> LoteTransiciones:
        """Devuelve los pares acumulados desde iniciar_bloque y los descarta.

        Igual que extraer_bloque, sin las recompensas.

        Raises:
            ValueError: Si no se ha dado ningún paso desde iniciar_bloque.
        """
        return self.extraer_bloque()[0]


def _retornos_gae(recompensas: np.ndarray, valores: np.ndarray, inicios_episodio: np.ndarray,
                  ultimos_valores: np.ndarray, ultimos_terminados: np.ndarray,
                  gamma: float, lam: float) -> np.ndarray:
    """Calcula los retornos TD(lambda) de un bloque con la fórmula de sb3.

    Reproduce RolloutBuffer.compute_returns_and_advantage: ventajas GAE(lambda)
    recorriendo el bloque hacia atrás y retornos = ventajas + valores. Los
    arrays son [pasos, entornos], sin aplanar; ultimos_valores y
    ultimos_terminados son del estado que sigue al último paso [entornos].
    """
    pasos = len(recompensas)
    retornos = np.zeros(recompensas.shape, dtype=np.float32)
    ventaja = np.zeros(recompensas.shape[1], dtype=np.float32)
    for paso in reversed(range(pasos)):
        if paso == pasos - 1:
            no_terminal = 1.0 - ultimos_terminados.astype(np.float32)
            siguientes = ultimos_valores
        else:
            no_terminal = 1.0 - inicios_episodio[paso + 1]
            siguientes = valores[paso + 1]
        delta = recompensas[paso] + gamma * siguientes * no_terminal - valores[paso]
        ventaja = delta + gamma * lam * no_terminal * ventaja
        retornos[paso] = ventaja + valores[paso]
    return retornos


def _recompensas_de_bloque(valor: object, nombre: str, forma: tuple[int, int]) -> np.ndarray:
    """Comprueba que unas recompensas son reales, finitas y de la forma del bloque."""
    array = np.asarray(valor)
    if array.shape != forma or array.dtype.kind not in "iuf":
        raise ValueError(f"{nombre} deben ser números reales {list(forma)} [pasos, entornos], "
                         f"no {array.dtype} {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{nombre} deben ser finitas")
    return array.astype(np.float32)


def reajustar_valor_tras_discriminador(agente: TRPO, recompensas_anteriores: np.ndarray,
                                       recompensas_nuevas: np.ndarray) -> dict[str, float]:
    """Pone la red de valor al día con el discriminador recién actualizado.

    El código original de GAIL, después de actualizar el discriminador, vuelve
    a puntuar el lote con el discriminador nuevo y reajusta la red de valor con
    esas recompensas, para que la siguiente iteración calcule las ventajas con
    una red que corresponde al discriminador con el que se va a jugar. TRPO de
    sb3 ajusta la red de valor dentro de learn(), antes de que se actualice el
    discriminador, así que sin este paso iría siempre un discriminador por
    detrás (decisión del 14 de octubre).

    Se llama una vez por bloque, justo después de agente.learn() con un único
    bloque de recogida y de actualizar el discriminador. Usa el rollout_buffer
    que sb3 conserva hasta la siguiente recogida:

    1. Suma a las recompensas del buffer la diferencia nuevas − anteriores. Así
       se conserva el término gamma · V(estado final) que sb3 añade a las
       partidas truncadas por límite de tiempo.
    2. Recalcula los retornos con GAE, con gamma y gae_lambda del agente y los
       valores de la red de valor actual.
    3. Repite el ajuste de la red de valor de TRPO.train(): n_critic_updates
       pasadas por minibatches de batch_size, error cuadrático y el optimizador
       de la política, sin mover los parámetros de la política.

    Diferencias con el original: los objetivos son retornos GAE(lambda), como
    los que usa sb3 en cada bloque, y no retornos de Monte Carlo; y el ajuste
    es el de sb3 (Adam por minibatches). Depende de atributos internos de sb3
    (rollout_buffer, _last_obs, _last_episode_starts), válidos con las
    versiones fijadas en requirements.txt. El orden de los minibatches sale del
    generador global de NumPy, como en TRPO.train().

    Args:
        agente: TRPO con política MLP y observaciones de tipo array, tras learn().
        recompensas_anteriores: Las que dio el discriminador durante el bloque,
            [n_steps, n_envs] (EntornoRecompensaGAIL.extraer_bloque).
        recompensas_nuevas: Las de los mismos pares con el discriminador
            actualizado, con la misma forma.

    Returns:
        {"perdida_valor": media del error cuadrático de los minibatches}; nan
        si n_critic_updates es 0.

    Raises:
        ValueError: Si el agente no es TRPO, su buffer no tiene un bloque recién
            entrenado, ya se reajustó en este bloque o las recompensas no son
            válidas.
    """
    if not isinstance(agente, TRPO):
        raise ValueError(f"Se esperaba un agente TRPO de sb3-contrib, no {type(agente).__name__}")
    buffer = agente.rollout_buffer
    if type(buffer) is not RolloutBuffer:
        raise ValueError("Solo se admite el RolloutBuffer de observaciones de tipo array")
    if not buffer.full or not buffer.generator_ready or agente._last_obs is None:
        raise ValueError("El buffer no tiene un bloque entrenado: llamar justo después de learn()")
    # Aplicar dos veces la diferencia la sumaría dos veces al mismo bloque.
    if getattr(agente, "_pasos_ultimo_reajuste", None) == agente.num_timesteps:
        raise ValueError("Este bloque ya se reajustó: solo se puede reajustar una vez por bloque")
    forma = (buffer.buffer_size, buffer.n_envs)
    anteriores = _recompensas_de_bloque(recompensas_anteriores, "recompensas_anteriores", forma)
    nuevas = _recompensas_de_bloque(recompensas_nuevas, "recompensas_nuevas", forma)

    buffer.rewards += nuevas - anteriores
    politica = agente.policy
    with torch.no_grad():
        valores_planos = politica.predict_values(
            obs_as_tensor(buffer.observations, agente.device)).cpu().numpy()
        ultimos_valores = politica.predict_values(
            obs_as_tensor(agente._last_obs, agente.device)).cpu().numpy().flatten()
    # El buffer guarda los datos aplanados entorno a entorno: índice = entorno · pasos + paso.
    valores = valores_planos.reshape(buffer.n_envs, buffer.buffer_size).T
    retornos = _retornos_gae(buffer.rewards, valores, buffer.episode_starts, ultimos_valores,
                             np.asarray(agente._last_episode_starts), buffer.gamma,
                             buffer.gae_lambda)
    buffer.values = buffer.swap_and_flatten(valores.astype(np.float32))
    buffer.returns = buffer.swap_and_flatten(retornos)
    buffer.advantages = buffer.returns - buffer.values
    agente._pasos_ultimo_reajuste = agente.num_timesteps

    modo_previo = politica.training
    politica.set_training_mode(True)
    perdidas = []
    try:
        for _ in range(agente.n_critic_updates):
            for datos in buffer.get(agente.batch_size):
                prediccion = politica.predict_values(datos.observations)
                perdida = F.mse_loss(datos.returns, prediccion.flatten())
                politica.optimizer.zero_grad()
                perdida.backward()
                # Como TRPO.train(): el optimizador cubre toda la política, pero
                # solo se mueven los parámetros de la red de valor.
                for nombre, parametro in politica.named_parameters():
                    if "value" not in nombre:
                        parametro.grad = None
                politica.optimizer.step()
                perdidas.append(perdida.item())
    finally:
        politica.set_training_mode(modo_previo)
    return {"perdida_valor": float(np.mean(perdidas)) if perdidas else float("nan")}



def _comprobar_piezas_iteracion(agente: TRPO, envoltorio: EntornoRecompensaGAIL,
                                optimizador: torch.optim.Optimizer,
                                pares_expertos: LoteTransiciones,
                                generador: np.random.Generator, cfg: dict) -> None:
    """Comprueba antes de jugar que las piezas de iteracion_gail encajan entre sí.

    Raises:
        ValueError: Si alguna pieza no es válida o no corresponde a las demás.
    """
    if not isinstance(agente, TRPO):
        raise ValueError(f"Se esperaba un agente TRPO de sb3-contrib, no {type(agente).__name__}")
    if not isinstance(envoltorio, EntornoRecompensaGAIL):
        raise ValueError("envoltorio debe ser un EntornoRecompensaGAIL")
    # El agente juega en envoltorio, directamente o con VecNormalize encima. Si
    # algo encima cambiara la recompensa, el buffer no guardaría la del discriminador.
    capa = agente.get_env()
    while capa is not envoltorio:
        if isinstance(capa, VecNormalize) and capa.norm_reward:
            raise ValueError("VecNormalize con norm_reward cambiaría la recompensa del "
                             "discriminador: solo se admite normalizar observaciones")
        capa = capa.venv if isinstance(capa, VecEnvWrapper) else None
        if capa is None:
            raise ValueError("El agente no juega en este envoltorio")
    parametros = {id(p) for p in envoltorio.discriminador.parameters()}
    del_optimizador = {id(p) for grupo in optimizador.param_groups for p in grupo["params"]}
    if del_optimizador != parametros:
        raise ValueError("El optimizador no es el del discriminador del envoltorio")
    _config_discriminador(cfg)
    _comprobar_lote(pares_expertos, "pares_expertos")
    estados, _ = tensores_de_pares(pares_expertos.estados, pares_expertos.acciones,
                                   envoltorio.action_space)
    if estados.shape[1] != envoltorio.discriminador.dim_observacion:
        raise ValueError("Los pares expertos no tienen la dimensión de observación del entorno")
    if not isinstance(generador, np.random.Generator):
        raise ValueError("generador debe ser un numpy.random.Generator de crear_generador")


def iteracion_gail(agente: TRPO, envoltorio: EntornoRecompensaGAIL,
                   optimizador: torch.optim.Optimizer, pares_expertos: LoteTransiciones,
                   generador: np.random.Generator, cfg: dict) -> dict[str, float]:
    """Da una iteración de GAIL: un bloque de TRPO, el discriminador y la red de valor.

    1. Juega un bloque de TRPO (iteracion_trpo) en el que cada paso recibe la
       recompensa del discriminador actual, que está fijo durante el bloque.
    2. Actualiza el discriminador con los pares del bloque frente a otros
       tantos pares expertos muestreados con reemplazo.
    3. Vuelve a puntuar los pares del bloque con el discriminador actualizado y
       reajusta la red de valor con esas recompensas
       (reajustar_valor_tras_discriminador).

    Todo se comprueba antes de jugar, así que un error en los argumentos no
    deja la iteración a medias.

    Args:
        agente: TRPO creado sobre envoltorio, directamente o con VecNormalize
            encima (solo de observaciones: el discriminador ve las originales).
        envoltorio: El EntornoRecompensaGAIL en el que juega el agente; su
            discriminador es el que se actualiza.
        optimizador: El del discriminador (preparar_discriminador); se conserva
            entre iteraciones.
        pares_expertos: Pares expertos del subconjunto de train.
        generador: Generador de crear_generador; elige los pares expertos.
        cfg: Configuración resuelta con la sección gail.

    Returns:
        Medidas de la iteración: las de iteracion_trpo (pasos, partidas,
        retorno_real_medio, longitud_media), recompensa_gail_media (la que
        recibió el agente en el bloque, sin bootstrap), las de
        actualizar_discriminador con el prefijo discriminador/ y
        reajuste/perdida_valor.

    Raises:
        ValueError: Si alguna pieza no es válida o no corresponde a las demás.
    """
    _comprobar_piezas_iteracion(agente, envoltorio, optimizador, pares_expertos, generador, cfg)
    discriminador, espacio_accion = envoltorio.discriminador, envoltorio.action_space
    envoltorio.iniciar_bloque()
    medidas = iteracion_trpo(agente)
    lote, anteriores = envoltorio.extraer_bloque()
    expertos = muestrear_expertos(pares_expertos, lote.numero_transiciones, generador)
    medidas_discriminador = actualizar_discriminador(discriminador, optimizador, lote, expertos,
                                                     espacio_accion, cfg)
    # El lote va paso a paso y, dentro de cada paso, entorno a entorno: el mismo
    # orden que las recompensas [pasos, entornos].
    nuevas = calcular_recompensas_gail(discriminador, lote.estados, lote.acciones,
                                       espacio_accion, envoltorio.modo).reshape(anteriores.shape)
    reajuste = reajustar_valor_tras_discriminador(agente, anteriores, nuevas)
    medidas["recompensa_gail_media"] = float(anteriores.mean())
    medidas.update({f"discriminador/{clave}": valor
                    for clave, valor in medidas_discriminador.items()})
    medidas["reajuste/perdida_valor"] = reajuste["perdida_valor"]
    return medidas


def _config_gail(cfg: dict) -> dict:
    """Devuelve la sección gail comprobada (sin el discriminador, que se comprueba aparte).

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    gcfg = cfg.get("gail") if isinstance(cfg, dict) else None
    if not isinstance(gcfg, dict):
        raise ValueError("La configuración necesita la sección gail")

    reglas = {
        "iteraciones": es_entero_positivo,
        "evaluar_cada": es_entero_positivo,
        "episodios_evaluacion": es_entero_positivo,
        "evaluacion_determinista": lambda v: isinstance(v, bool),
        "recompensa": lambda v: v in MODOS_RECOMPENSA,
    }
    for clave, valida in reglas.items():
        if clave not in gcfg:
            raise ValueError(f"Falta gail.{clave} en la configuración")
        if not valida(gcfg[clave]):
            raise ValueError(f"gail.{clave} no es válido: {gcfg[clave]!r}")
    return gcfg


def comprobar_config_barrido(cfg: dict) -> None:
    """Comprueba la configuración del barrido de GAIL antes de entrenar nada.

    Reúne lo que cada entrenamiento comprobaría por su cuenta (secciones gail,
    gail.discriminador y trpo) y lo común a los barridos (comprobar_config_comun:
    demostraciones y experimento). Así scripts/entrenar_gail.py rechaza una
    configuración no válida una sola vez y sin lanzar procesos, en lugar de
    fallar en cada entrenamiento.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    _config_gail(cfg)
    _config_discriminador(cfg)
    config_trpo(cfg)
    comprobar_config_comun(cfg)


def entrenar_gail(cfg: dict, trayectorias: Sequence[Trayectoria], splits: SplitsDemostraciones,
                  cantidad: int, repeticion: int, run: Any = None,
                  ) -> tuple[TRPO, Discriminador, dict[str, Any]]:
    """Entrena GAIL con un subconjunto de partidas expertas, en una repetición.

    Las semillas son las de la repetición (semillas_repeticion con
    experimento.semilla_repeticiones), compartidas con BC:

    - pares: elige los pares expertos de cada partida con pares_de_subconjunto,
      como BC (los mismos pares de train con la misma repetición), y después
      los que se muestrean en cada iteración;
    - modelo: fijar_seed antes de crear el discriminador y TRPO;
    - entornos: los entornos de entrenamiento;
    - evaluacion: las partidas de validación.

    Monta el discriminador (preparar_discriminador, con la normalización de
    los pares expertos de train), el envoltorio, VecNormalize encima si el
    perfil normaliza las observaciones y TRPO (crear_trpo), y da
    gail.iteraciones iteraciones (iteracion_gail). Cada gail.evaluar_cada
    iteraciones, y en la última, juega gail.episodios_evaluacion partidas de
    validación con la recompensa real (siempre las mismas semillas). Ese
    retorno solo se registra: el modelo que se devuelve es el de la última
    iteración, como en el paper. Las partidas de validación no alteran el
    entrenamiento.

    Args:
        cfg: Configuración resuelta con entorno, trpo, gail, demostraciones y
            experimento.semilla_repeticiones.
        trayectorias: Todas las partidas del dataset para el que se crearon los splits.
        splits: Splits de esas partidas.
        cantidad: Tamaño del subconjunto de train, una clave de splits.subconjuntos.
        repeticion: Número de repetición, entero no negativo.
        run: Run de W&B o None; recibe las medidas de cada iteración y las de
            validación (prefijo validacion/), con el paso de entorno acumulado.

    Returns:
        El agente, el discriminador y un resumen: entorno, cantidad, repeticion,
        IDs de las partidas, número de pares expertos, huella del dataset,
        semillas, iteraciones, pasos, medidas de cada iteración, validaciones,
        segundos medidos y estadísticas de normalización (None si no normaliza).

    Raises:
        ValueError: Si la configuración, los datos o los argumentos no son
            válidos. Todo se comprueba antes de crear los entornos salvo que
            los pares encajen con sus espacios, que se comprueba antes de jugar.
    """
    gcfg, tcfg = _config_gail(cfg), config_trpo(cfg)
    _config_discriminador(cfg)
    ecfg = cfg.get("experimento") if isinstance(cfg, dict) else None
    if not isinstance(ecfg, dict) or "semilla_repeticiones" not in ecfg:
        raise ValueError("La configuración necesita experimento.semilla_repeticiones")
    semillas = semillas_repeticion(ecfg["semilla_repeticiones"], repeticion)
    generador = crear_generador(semillas["pares"])
    _, pares = pares_de_subconjunto(trayectorias, splits, cantidad, cfg, generador)

    fijar_seed(semillas["modelo"])
    venv: VecEnv = crear_entorno_vectorizado(cfg, tcfg["n_entornos"], semillas["entornos"])
    inicio = time.perf_counter()
    try:
        discriminador, optimizador = preparar_discriminador(
            cfg, venv.observation_space, venv.action_space, pares)
        envoltorio = EntornoRecompensaGAIL(venv, discriminador, gcfg["recompensa"])
        capa: VecEnv = envoltorio
        if tcfg["normalizar_observaciones"]:
            capa = VecNormalize(envoltorio, norm_obs=True, norm_reward=False,
                                clip_obs=float(tcfg["limite_normalizacion"]))
        agente = crear_trpo(cfg, capa)
        historial, validaciones = [], []
        for iteracion in range(1, gcfg["iteraciones"] + 1):
            medidas = iteracion_gail(agente, envoltorio, optimizador, pares, generador, cfg)
            registrar(run, con_seccion(medidas, SECCION_TRPO), paso=agente.num_timesteps)
            historial.append({"iteracion": iteracion, **medidas})
            if iteracion % gcfg["evaluar_cada"] and iteracion < gcfg["iteraciones"]:
                continue
            validacion = evaluar_retorno(agente, cfg, gcfg["episodios_evaluacion"],
                                         semillas["evaluacion"],
                                         gcfg["evaluacion_determinista"],
                                         estadisticas_normalizacion(capa))
            validacion.update(iteracion=iteracion, pasos=agente.num_timesteps)
            validaciones.append(validacion)
            registrar(run, {f"validacion/{clave}": validacion[clave] for clave in
                            ("retorno_medio", "retorno_desviacion", "longitud_media")},
                      paso=agente.num_timesteps)
        segundos = time.perf_counter() - inicio
        normalizacion = estadisticas_normalizacion(capa)
    finally:
        venv.close()
    resumen = {
        "entorno": cfg["entorno"]["id"],
        "cantidad": int(cantidad),
        "repeticion": int(repeticion),
        "ids_partidas": [int(i) for i in splits.subconjuntos[cantidad]],
        "pares_expertos": pares.numero_transiciones,
        "huella_dataset": splits.huella_dataset,
        "semillas": {"base": int(ecfg["semilla_repeticiones"]), **semillas},
        "iteraciones": gcfg["iteraciones"],
        "pasos": int(agente.num_timesteps),
        "historial": historial,
        "validaciones": validaciones,
        "segundos": segundos,
        "normalizacion": normalizacion,
    }
    return agente, discriminador, resumen


def _para_json(valor: object) -> object:
    """Copia valor con tipos de JSON: escalares de NumPy a Python y nan o infinitos a None.

    Raises:
        TypeError: Si hay algo que no se puede guardar en JSON.
    """
    if isinstance(valor, dict):
        return {clave: _para_json(v) for clave, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_para_json(v) for v in valor]
    if not (valor is None or isinstance(valor, (str, int, float, bool))):
        valor = a_json(valor)  # escalares de NumPy; TypeError con cualquier otra cosa
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def guardar_gail(directorio: Path, agente: TRPO, discriminador: Discriminador, cfg: dict,
                 resumen: dict[str, Any], datos: dict[str, str]) -> None:
    """Guarda un entrenamiento de GAIL en un directorio nuevo, como el experto.

    - modelo.zip: el agente TRPO (agente.save).
    - discriminador.pt: el state_dict del discriminador (pesos y
      normalización), sin el optimizador.
    - gail.json: la configuración resuelta, el resumen de entrenar_gail
      (cantidad, repetición, IDs de las partidas, semillas, medidas,
      validaciones, tiempo y normalización), la procedencia de los datos
      (procedencia_datos), los argumentos para reconstruir el discriminador
      (leídos del propio discriminador, activación incluida), el commit de
      git y si había cambios sin commit, las versiones y el hardware. Es JSON
      estricto: las medidas no definidas (nan, por ejemplo
      el retorno de una iteración en la que no terminó ninguna partida) se
      guardan como null.

    No sobrescribe (comprobar_directorio_libre) y lo comprueba todo antes de
    crear la carpeta. Escribe con escribir_carpeta_nueva: si falla al escribir,
    no queda una carpeta a medias.

    Raises:
        FileExistsError: Si el directorio existe con contenido.
        ValueError: Si las piezas no son válidas, los datos no son aquellos con
            los que se entrenó (huella del NPZ distinta de la del resumen) o el
            discriminador no corresponde a la configuración (capas ocultas o
            activación distintas de gail.discriminador).
    """
    if not isinstance(agente, TRPO):
        raise ValueError("agente debe ser un TRPO de sb3-contrib")
    if not isinstance(discriminador, Discriminador):
        raise ValueError("discriminador debe ser un Discriminador")
    claves_datos = {"demostraciones", "splits", "huella_dataset", "huella_splits",
                    "huella_partidas"}
    if not isinstance(datos, dict) or set(datos) != claves_datos:
        raise ValueError("datos debe ser el diccionario de procedencia_datos")
    if not isinstance(resumen, dict) or datos["huella_dataset"] != resumen.get("huella_dataset"):
        raise ValueError("Las demostraciones no son las del entrenamiento: la huella del NPZ "
                         "no coincide con la de los splits usados")
    # La arquitectura se lee del propio discriminador: la activación no tiene pesos,
    # así que comparar el state_dict con la configuración no la comprobaba.
    argumentos = {"dim_observacion": discriminador.dim_observacion,
                  "dim_accion": discriminador.dim_accion, "continua": discriminador.continua,
                  "capas_ocultas": list(discriminador.capas_ocultas),
                  "activacion": discriminador.activacion}
    dcfg = cfg.get("gail", {}).get("discriminador", {}) if isinstance(cfg, dict) else {}
    pedida = {"capas_ocultas": dcfg.get("capas_ocultas"), "activacion": dcfg.get("activacion")}
    if (not isinstance(pedida["capas_ocultas"], (list, tuple))
            or list(pedida["capas_ocultas"]) != argumentos["capas_ocultas"]
            or pedida["activacion"] != argumentos["activacion"]):
        raise ValueError(
            "El discriminador no corresponde a gail.discriminador de la configuración: tiene "
            f"capas {argumentos['capas_ocultas']} y activación {argumentos['activacion']!r}, "
            f"y la configuración pide {pedida['capas_ocultas']} y {pedida['activacion']!r}")
    directorio = comprobar_directorio_libre(directorio)
    metadatos = {"configuracion": cfg, "resumen": resumen, "datos": datos,
                 "discriminador": argumentos, "codigo": estado_git(),
                 "entorno_ejecucion": versiones_y_hardware()}
    # Se serializa antes de crear nada: un valor no válido no deja una carpeta a medias.
    texto = json.dumps(_para_json(metadatos), ensure_ascii=False, indent=2, allow_nan=False)

    def escribir(carpeta: Path) -> None:
        """Escribe gail.json, discriminador.pt y modelo.zip en carpeta."""
        with (carpeta / _ARCHIVO_METADATOS).open("x", encoding="utf-8") as archivo:
            archivo.write(texto)
        torch.save(discriminador.state_dict(), carpeta / _ARCHIVO_DISCRIMINADOR)
        agente.save(carpeta / _ARCHIVO_MODELO)

    escribir_carpeta_nueva(directorio, escribir)


def cargar_gail(directorio: Path) -> tuple[TRPO, Discriminador, dict[str, Any]]:
    """Carga un entrenamiento guardado con guardar_gail, en CPU.

    El discriminador se reconstruye con los argumentos de gail.json y se carga
    con weights_only=True (solo tensores, sin ejecutar código del archivo); se
    deja en modo evaluación.

    Returns:
        El agente, el discriminador y los metadatos; las estadísticas de
        normalización están en metadatos["resumen"]["normalizacion"] (None si
        no normaliza), para politica_trpo.

    Raises:
        FileNotFoundError: Si falta alguno de los tres archivos.
        ValueError: Si el discriminador guardado no corresponde a sus argumentos.
    """
    directorio = Path(directorio)
    for nombre in (_ARCHIVO_MODELO, _ARCHIVO_DISCRIMINADOR, _ARCHIVO_METADATOS):
        if not (directorio / nombre).is_file():
            raise FileNotFoundError(f"Falta {nombre} en {directorio}")
    with (directorio / _ARCHIVO_METADATOS).open(encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    discriminador = Discriminador(**metadatos["discriminador"])
    estado = torch.load(directorio / _ARCHIVO_DISCRIMINADOR, map_location="cpu",
                        weights_only=True)
    try:
        discriminador.load_state_dict(estado)
    except RuntimeError as error:
        raise ValueError(f"discriminador.pt no corresponde a gail.json: {error}") from error
    discriminador.eval()
    return TRPO.load(directorio / _ARCHIVO_MODELO, device="cpu"), discriminador, metadatos


def entrenar_y_guardar_gail(cfg: dict, trayectorias: Sequence[Trayectoria],
                            splits: SplitsDemostraciones, cantidad: int, repeticion: int,
                            destino: Path, datos: dict[str, str],
                            grupo: str | None = None) -> dict[str, Any]:
    """Entrena un GAIL del barrido y lo guarda: la tarea de cada proceso del script.

    Abre su propio run de W&B (según wandb.modo, con la cantidad y la
    repetición en el nombre, provisional), imprime el progreso con
    RegistroProgreso, entrena con entrenar_gail y guarda con guardar_gail. El
    run se cierra con terminar_run: como fallido si falla el entrenamiento o
    el guardado.
    Solo depende de sus argumentos, así que da lo mismo en serie que en
    paralelo, en otro proceso.

    Args:
        cfg: Configuración resuelta.
        trayectorias: Todas las partidas del dataset.
        splits: Sus splits.
        cantidad: Tamaño del subconjunto de train.
        repeticion: Número de repetición.
        destino: Carpeta nueva donde guardarlo.
        datos: Procedencia de los datos (procedencia_datos).

    Returns:
        El resumen de entrenar_gail.

    Raises:
        ValueError: Si algo no es válido (ver entrenar_gail y guardar_gail).
        FileExistsError: Si destino existe con contenido.
    """
    run = abrir_run_barrido(cfg, "gail", cantidad, repeticion, grupo)
    progreso = RegistroProgreso(run, f"cantidad {cantidad}, repetición {repeticion}",
                                cfg["trpo"]["pasos_por_iteracion"], cfg["gail"]["iteraciones"])
    try:
        if cfg.get("wandb", {}).get("modo") == "online":
            usar_artefacto(run, nombre_artefacto("demostraciones", datos["huella_dataset"])
                           + ":latest")
        agente, discriminador, resumen = entrenar_gail(cfg, trayectorias, splits, cantidad,
                                                       repeticion, progreso)
        guardar_gail(destino, agente, discriminador, cfg, resumen, datos)
        registrar_resumen(run, {**resumen, **datos})
        if run is not None:
            huella_modelo = huella_archivo(destino / "modelo.zip")
            subir_artefacto(run, nombre_artefacto("gail", huella_modelo, cfg["entorno"]["id"],
                                                  cantidad, repeticion), "gail",
                            [destino / "modelo.zip", destino / "discriminador.pt",
                             destino / "gail.json"],
                            {"cantidad": cantidad, "repeticion": repeticion,
                             "huella_modelo": huella_modelo, **datos})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    return resumen


def _describir_gail(resumen: dict[str, Any]) -> str:
    """Resume la última validación, los pasos y el tiempo de entrenamiento."""
    ultima = resumen["validaciones"][-1]
    return (f"retorno de validación {ultima['retorno_medio']:.1f} en la iteración "
            f"{ultima['iteracion']}; {resumen['pasos']} pasos en {resumen['segundos']:.0f} s")


METODO_GAIL = MetodoBarrido(
    nombre="gail", comprobar_config=comprobar_config_barrido,
    tarea=entrenar_y_guardar_gail,
    destino=lambda salida, c, r: salida / f"cantidad_{c}" / f"repeticion_{r}",
    describir=_describir_gail,
)
