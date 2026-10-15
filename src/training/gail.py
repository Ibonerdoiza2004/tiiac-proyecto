"""Recompensa de GAIL: discriminador, su entrenamiento y el envoltorio del entorno.

Sigue el código original de GAIL (openai/imitation, TransitionClassifier):

- Durante un bloque de recogida el discriminador está fijo y en modo
  evaluación; cada paso recibe la recompensa que calcula a partir del par
  (estado, acción) en lugar de la del entorno (EntornoRecompensaGAIL).
- Después del bloque se actualiza con los pares del agente frente a otros
  tantos pares expertos muestreados con reemplazo (muestrear_expertos y
  actualizar_discriminador).

El bucle que alterna bloques de TRPO y actualizaciones vive en la rama gail.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from src.data.loaders import tensores_de_pares
from src.data.tipos import LoteTransiciones
from src.models.factory import Discriminador, construir_discriminador

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
    claves = {"optimizador", "learning_rate", "pasos_adam", "peso_entropia",
              "normalizar_entrada", "epsilon_normalizacion"}
    if not isinstance(dcfg, dict) or not claves <= dcfg.keys():
        raise ValueError("La configuración necesita gail.discriminador con "
                         + ", ".join(sorted(claves)))

    def real(clave: str) -> float:
        """Devuelve dcfg[clave] si es un real finito no booleano."""
        valor = dcfg[clave]
        if (isinstance(valor, bool) or not isinstance(valor, (int, float))
                or not math.isfinite(valor)):
            raise ValueError(f"gail.discriminador.{clave} debe ser un real finito, no {valor!r}")
        return float(valor)

    if dcfg["optimizador"] != "adam":
        raise ValueError(f"gail.discriminador.optimizador desconocido: {dcfg['optimizador']!r}; "
                         "opciones: ['adam']")
    if real("learning_rate") <= 0:
        raise ValueError("gail.discriminador.learning_rate debe ser positivo")
    pasos = dcfg["pasos_adam"]
    if isinstance(pasos, bool) or not isinstance(pasos, int) or pasos <= 0:
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
    if isinstance(cantidad, (bool, np.bool_)) or not isinstance(cantidad, (int, np.integer)) \
            or cantidad <= 0:
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
        self._acciones = np.array(actions, copy=True)
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        """Da el paso y sustituye la recompensa por la del discriminador."""
        observaciones, recompensas_originales, terminados, infos = self.venv.step_wait()
        recompensas = calcular_recompensas_gail(self.discriminador, self._observaciones,
                                                self._acciones, self.action_space, self.modo)
        for info, original in zip(infos, recompensas_originales, strict=True):
            info["recompensa_original"] = float(original)
        self._estados_bloque.append(self._observaciones)
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
        """Descarta los pares acumulados para empezar un bloque nuevo."""
        self._estados_bloque: list[np.ndarray] = []
        self._acciones_bloque: list[np.ndarray] = []
        self._ids_bloque: list[np.ndarray] = []
        self._pasos_bloque: list[np.ndarray] = []

    def extraer_lote(self) -> LoteTransiciones:
        """Devuelve los pares acumulados desde iniciar_bloque y los descarta.

        Raises:
            ValueError: Si no se ha dado ningún paso desde iniciar_bloque.
        """
        if not self._estados_bloque:
            raise ValueError("No hay pares: no se ha dado ningún paso en el bloque")
        lote = LoteTransiciones(np.concatenate(self._estados_bloque),
                                np.concatenate(self._acciones_bloque),
                                np.concatenate(self._ids_bloque),
                                np.concatenate(self._pasos_bloque))
        self.iniciar_bloque()
        return lote
