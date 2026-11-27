"""Contrato de la API: los modelos de todas las peticiones, respuestas y mensajes.

Es la única definición del contrato entre el backend y el frontend. FastAPI genera
con estos modelos la documentación de /docs, y los de los mensajes del WebSocket, que
OpenAPI no recoge, sirven para validar lo que llega del navegador y como referencia.

Convención (protocolo único):

- Cada funcionalidad tiene su ruta para *crear* (``POST /api/sesiones``,
  ``POST /api/entrenamientos``, ``POST /api/comparaciones``), porque pide datos
  distintos y elige qué código se ejecuta. Todas devuelven un ``TrabajoApi``, cuyos
  ``argumentos`` son los públicos que fija esa ruta.
- Todo lo demás es común a cualquier trabajo: consultar, pausar, reanudar, cancelar
  (``/api/trabajos/{id}``) y seguirlo por el WebSocket ``/api/trabajos/{id}``.
- Solo hay rutas propias donde cambia el significado, como guardar una sesión.

Mensajes del servidor por el WebSocket, todos con ``tipo``: ``estado``,
``historial``, ``progreso``, ``error``, imágenes en binario y los eventos propios de
cada funcionalidad (``partida_terminada`` en las sesiones). Mensajes del navegador:
``pausar``, ``reanudar``, ``cancelar`` y las órdenes de cada tipo de trabajo
(ORDENES_POR_TIPO).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from src.app.trabajos import EstadoTrabajo
from src.utils.configuracion import TipoEntorno

MetodoEntrenamiento = Literal["experto", "bc", "gail"]
ModoSesion = Literal["manual", "experto"]
TipoCampo = Literal["entero", "real", "booleano", "opcion", "lista_enteros"]
NivelCampo = Literal["basico", "avanzado"]
UnidadDuracion = Literal["pasos", "epocas", "iteraciones"]
Natural = Annotated[int, Field(strict=True, ge=0)]
Positivo = Annotated[int, Field(strict=True, ge=1)]
# Estrictos: un booleano o un entero del navegador tiene que serlo de verdad («true» o
# 1 no valen como booleano, ni «3» como entero).
Booleano = Annotated[bool, Field(strict=True)]
# Nombre que pone el usuario a algo de la biblioteca: con algún carácter visible.
NombreUsuario = Annotated[str, Field(min_length=1, max_length=100, pattern=r"\S")]
# Grupo de W&B: letras, números, guiones, puntos y guiones bajos.
GrupoWandb = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")]


class _Estricto(BaseModel):
    """Base de las peticiones: rechaza claves que no estén en el modelo."""

    model_config = ConfigDict(extra="forbid")


# --- General ----------------------------------------------------------------------

class EntornoApi(BaseModel):
    """Un entorno del dashboard: la clave de su perfil y su id de Gymnasium."""

    id: str
    gym: str


class WandbApi(BaseModel):
    """Si se puede registrar en W&B (hay sesión válida)."""

    disponible: bool | None = Field(description="None mientras se comprueba al arrancar.")
    motivo: str | None = Field(description="Por qué no está disponible.")


class EstadoApi(BaseModel):
    """Respuesta de GET /api/estado."""

    entornos: list[EntornoApi]
    procesos: int
    libres: int
    wandb: WandbApi


class TrabajoApi(BaseModel):
    """Un trabajo (sesión, entrenamiento o comparación): lo que devuelven las rutas de
    crear y las de /api/trabajos."""

    id: str
    tipo: str
    estado: EstadoTrabajo
    argumentos: dict[str, Any]
    interactivo: bool
    creado: float
    iniciado: float | None
    terminado: float | None
    progresos: int
    resultado: dict[str, Any] | None
    error: str | None


# --- Mensajes del servidor por el WebSocket ---------------------------------------------

class MensajeEstado(BaseModel):
    """Al conectar y en cada cambio de estado; el último lleva el resultado."""

    tipo: Literal["estado"]
    trabajo: TrabajoApi


class MensajeHistorial(BaseModel):
    """Al conectar: todo el progreso y los eventos propios hasta ese momento."""

    tipo: Literal["historial"]
    progreso: list[dict[str, Any]]
    eventos: list[dict[str, Any]]


class MensajeProgreso(BaseModel):
    """En directo: medidas de una iteración, época o paso de la sesión."""

    tipo: Literal["progreso"]
    medidas: dict[str, Any]


class MensajeError(BaseModel):
    """Un mensaje del navegador que no vale, o el trabajo no existe."""

    tipo: Literal["error"]
    mensaje: str


class PartidaApi(BaseModel):
    """Una partida terminada de una sesión."""

    indice: Natural
    pasos: Natural
    retorno: float
    semilla: Natural


class EventoPartidaTerminada(BaseModel):
    """Evento propio de las sesiones: acaba una partida."""

    tipo: Literal["partida_terminada"]
    partida: PartidaApi


class ResultadoSesion(BaseModel):
    """``resultado`` del estado final de una sesión: solo las partidas completas."""

    partidas: list[PartidaApi]


class ModeloGuardado(BaseModel):
    """Modelo que un entrenamiento deja en la biblioteca."""

    id: str
    nombre: str
    incompleto: bool


class ResultadoEntrenamiento(BaseModel):
    """``resultado`` del estado final de un entrenamiento."""

    modelo: ModeloGuardado
    semilla: Natural
    resumen: dict[str, Any]


class ModeloComparado(BaseModel):
    """Retornos de un modelo en una comparación (``aleatorio`` para el agente aleatorio)."""

    id: str
    retornos: list[float]


class ResultadoComparacion(BaseModel):
    """``resultado`` del estado final de una comparación: solo las rondas completas.

    ``partidas`` son las rondas que jugaron todos los modelos (cada uno tiene ese número
    de retornos) y ``semilla``, la de evaluacion en configs/. Incompleta si se finalizó
    antes de jugarlas todas.
    """

    partidas: Natural
    semilla: Natural
    incompleta: bool
    modelos: list[ModeloComparado]


# --- Mensajes del navegador por el WebSocket --------------------------------------------

class OrdenAccion(_Estricto):
    """Sesión manual: acción que se mantiene pulsada, o None sin ninguna tecla."""

    tipo: Literal["accion"]
    accion: Natural | None


class OrdenVista(_Estricto):
    """Entrenamiento: enciende o apaga el dibujo de las partidas."""

    tipo: Literal["vista"]
    activa: Booleano


class OrdenFinalizar(_Estricto):
    """Termina antes de tiempo guardando lo hecho (sesión: sus partidas completas;
    entrenamiento: el modelo de ese momento, marcado como incompleto). Si el trabajo
    está en pausa, se reanuda para poder terminar."""

    tipo: Literal["finalizar"]


# Controles que valen para cualquier trabajo; los atiende el gestor.
CONTROLES = ("pausar", "reanudar", "cancelar")
# Órdenes que acepta cada tipo de trabajo; se pasan al trabajo tal cual.
ORDENES_POR_TIPO: dict[str, tuple[type[BaseModel], ...]] = {
    "sesion": (OrdenAccion, OrdenFinalizar),
    "entrenamiento": (OrdenVista, OrdenFinalizar),
    "comparacion": (OrdenVista, OrdenFinalizar),
}
# Órdenes que reanudan el trabajo si está en pausa: un trabajo detenido no puede
# terminar, y se quedaría en pausa ocupando su hueco.
ORDENES_QUE_REANUDAN = frozenset({"finalizar"})


def validar_orden(tipo_trabajo: str, mensaje: Any) -> dict[str, Any]:
    """Comprueba que una orden del navegador vale para ese tipo de trabajo.

    Args:
        tipo_trabajo: Tipo del trabajo (sesion, entrenamiento, comparacion).
        mensaje: Mensaje ya decodificado, con su ``tipo``.

    Returns:
        La orden validada, como diccionario.

    Raises:
        ValueError: Si el tipo de orden no existe para ese trabajo o sus datos no valen.
    """
    tipo = mensaje.get("tipo") if isinstance(mensaje, dict) else None
    modelos = {m.model_fields["tipo"].annotation.__args__[0]: m
               for m in ORDENES_POR_TIPO.get(tipo_trabajo, ())}
    if tipo not in modelos:
        validas = ", ".join([*CONTROLES, *modelos]) or ", ".join(CONTROLES)
        raise ValueError(f"Orden «{tipo}» no válida para un trabajo de tipo {tipo_trabajo}; "
                         f"valen: {validas}")
    try:
        return TypeAdapter(modelos[tipo]).validate_python(mensaje).model_dump()
    except ValidationError as error:
        raise ValueError(f"Orden «{tipo}» no válida: {_resumir(error)}") from error


def _resumir(error: ValidationError) -> str:
    """Mensaje corto de un error de validación de Pydantic."""
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'mensaje'}: {e['msg']}"
                     for e in error.errors())


# --- Laboratorio: jugar y ver al experto -------------------------------------------------

class ConfiguracionJuegoApi(BaseModel):
    """Respuesta de GET /api/laboratorio/{entorno}."""

    entorno: TipoEntorno
    fps: Positivo = Field(description="Imágenes por segundo a velocidad ×1 (render_fps).")
    jugar_a_mano: bool
    partidas_experto: Positivo = Field(description="Partidas por defecto al grabar con el "
                                                   "experto (las del perfil).")
    partidas_maximas: Positivo
    semilla: Natural = Field(description="Semilla por defecto de las partidas "
                                         "(demostraciones.semilla).")


class SesionCrear(_Estricto):
    """Cuerpo de POST /api/sesiones: jugar a mano o ver al experto."""

    entorno: TipoEntorno
    modo: ModoSesion
    partidas: Positivo | None = Field(
        None, description="Solo en modo experto; por defecto, las del perfil. A mano se "
                          "juega hasta que el jugador finaliza.")
    experto: str | None = Field(
        None, description="Solo en modo experto: id de un experto de la biblioteca; por "
                          "defecto, el congelado del entorno.")
    semilla: Natural | None = Field(None, description="Por defecto, demostraciones.semilla.")

    @model_validator(mode="after")
    def _segun_modo(self) -> SesionCrear:
        """A mano no se piden partidas ni experto."""
        if self.modo == "manual" and (self.partidas is not None or self.experto is not None):
            raise ValueError("En modo manual no se indican partidas ni experto")
        return self


class SesionGuardar(_Estricto):
    """Cuerpo de POST /api/sesiones/{id}/guardar. Guardar cierra la sesión.

    ``partidas`` son los índices de las partidas terminadas que se guardan, enteras (no
    se recortan: decisión del usuario).
    """

    nombre: NombreUsuario
    partidas: Annotated[list[Natural], Field(min_length=1)]

    @model_validator(mode="after")
    def _partidas_distintas(self) -> SesionGuardar:
        """Cada partida, como mucho una vez."""
        if len(set(self.partidas)) != len(self.partidas):
            raise ValueError("Hay partidas repetidas")
        return self


class ResultadoGuardadoApi(BaseModel):
    """Respuesta de guardar: el conjunto nuevo de la biblioteca."""

    id: str
    nombre: str
    partidas: Positivo
    pares: Positivo


# --- Entrenamientos ------------------------------------------------------------------------

class CampoAjusteApi(BaseModel):
    """Un hiperparámetro del formulario."""

    clave: str
    tipo: TipoCampo
    valor: Any = Field(description="Valor del perfil.")
    etiqueta: str
    ayuda: str
    nivel: NivelCampo
    min: float | None = None
    max: float | None = None
    paso: float | None = None
    opciones: list[str] | None = None


class SeccionAjustesApi(BaseModel):
    """Una sección de configs/ (por ejemplo gail.discriminador) y sus campos."""

    clave: str
    titulo: str
    campos: list[CampoAjusteApi]


class DuracionApi(BaseModel):
    """Lo que mide el progreso y cuánto dura como mucho con los valores del perfil.

    En el experto son los pasos reales, no el presupuesto: TRPO da bloques completos y
    para en el primero que llega a experto.pasos_totales (pasos_reales_experto).
    ``campo`` y ``bloque`` dicen de qué campos del formulario sale, para que la interfaz
    la recalcule con lo que se edite: el total es el valor de ``campo`` o, con
    ``bloque``, ese valor redondeado hacia arriba a un múltiplo del de ``bloque``.
    """

    unidad: UnidadDuracion
    total: Positivo
    campo: str = Field(description="Campo del formulario (sección.clave) con la duración.")
    bloque: str | None = Field(None, description="Campo con el tamaño del bloque al que se "
                                                 "redondea (solo el experto).")


class EsquemaAjustesApi(BaseModel):
    """Respuesta de GET /api/laboratorio/{entorno}/ajustes/{metodo}."""

    metodo: MetodoEntrenamiento
    duracion: DuracionApi
    semilla: Natural = Field(description="Semilla por defecto (experto.semilla para el "
                                         "experto; experimento.semilla_repeticiones, "
                                         "repetición 0, para BC y GAIL).")
    secciones: list[SeccionAjustesApi]


class EntrenamientoCrear(_Estricto):
    """Cuerpo de POST /api/entrenamientos."""

    entorno: TipoEntorno
    metodo: MetodoEntrenamiento
    nombre: NombreUsuario | None = Field(
        None, description="Nombre del modelo en la biblioteca; si falta, se genera uno con "
                          "el método, el entorno y la fecha (p. ej. «GAIL · CartPole · "
                          "24/11 18:30»).")
    ajustes: dict[str, Any] = Field(
        default_factory=dict, description="Solo lo que cambia el usuario, anidado como en "
                                          "configs/, p. ej. {\"bc\": {\"learning_rate\": 0.01}}.")
    datos: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list, description="Ids de conjuntos y subconjuntos de la biblioteca "
                                          "(BC y GAIL); los pares repetidos cuentan una vez.")
    semilla: Natural | None = Field(None, description="Por defecto, la del esquema de ajustes.")
    vista: Booleano = Field(False, description="Dibujar las partidas (solo experto y GAIL).")
    wandb: Booleano = Field(False, description="Registrar también en W&B.")
    grupo: GrupoWandb | None = Field(
        None, description="Grupo de W&B (solo con wandb); por defecto, el de "
                          "configs/dashboard.yaml, «dashboard-<entorno>».")

    @model_validator(mode="after")
    def _segun_metodo(self) -> EntrenamientoCrear:
        """Datos solo para imitar; vista solo para los que juegan partidas."""
        if self.metodo == "experto" and self.datos:
            raise ValueError("El experto no usa demostraciones: datos debe ir vacío")
        if self.metodo != "experto" and not self.datos:
            raise ValueError(f"{self.metodo.upper()} necesita al menos un conjunto en datos")
        if len(set(self.datos)) != len(self.datos):
            raise ValueError("Hay conjuntos repetidos en datos")
        if self.metodo == "bc" and self.vista:
            raise ValueError("BC no juega partidas mientras entrena: no admite vista")
        if self.grupo is not None and not self.wandb:
            raise ValueError("El grupo de W&B solo tiene sentido con wandb: true")
        return self


# --- Comparar ---------------------------------------------------------------------------

IdBiblioteca = Annotated[str, Field(min_length=1)]


class ComparacionCrear(_Estricto):
    """Cuerpo de POST /api/comparaciones: modelos que juegan las mismas partidas.

    Se juega como la evaluación de la E1: semilla y modo de ``evaluacion`` en configs/,
    para que todas las comparaciones sean comparables entre sí.
    """

    entorno: TipoEntorno
    modelos: list[IdBiblioteca] = Field(
        default_factory=list, description="Ids de expertos, BC y GAIL de la biblioteca, del "
                                          "entorno y válidos, sin repetir.")
    aleatorio: Booleano = Field(False, description="Añadir el agente aleatorio como referencia.")
    partidas: Positivo | None = Field(
        None, description="Por modelo; por defecto, evaluacion.episodios (50). Como mucho "
                          "laboratorio.partidas_maximas de configs/dashboard.yaml.")
    vista: Booleano = Field(False, description="Dibujar las partidas, a la velocidad del "
                                               "entorno mientras está encendida.")

    @model_validator(mode="after")
    def _algo_que_comparar(self) -> ComparacionCrear:
        """Sin modelos repetidos y con al menos un agente."""
        if len(set(self.modelos)) != len(self.modelos):
            raise ValueError("Hay modelos repetidos")
        if not self.modelos and not self.aleatorio:
            raise ValueError("No hay nada que comparar: elige algún modelo o el aleatorio")
        return self


class ArgumentosComparacionApi(BaseModel):
    """``argumentos`` de un trabajo de comparación: la petición con lo que se usa de verdad.

    ``partidas`` y ``semilla`` son las efectivas (por defecto, las de ``evaluacion``).
    """

    entorno: TipoEntorno
    modelos: list[str]
    aleatorio: bool
    partidas: Positivo
    semilla: Natural
    vista: bool


class ProgresoComparacionApi(BaseModel):
    """``medidas`` de cada progreso de una comparación: una partida de un modelo.

    Todos los modelos juegan la partida i, con la misma semilla, antes de la i + 1.
    """

    modelo: str = Field(description="Id de la biblioteca o «aleatorio».")
    partida: Natural = Field(description="Desde 0.")
    retorno: float
    pasos: Positivo


# --- Biblioteca -------------------------------------------------------------------------

TipoElemento = Literal["experto", "bc", "gail", "demostraciones"]
OrigenElemento = Literal["congelado", "entrega", "dashboard"]


class ElementoBibliotecaApi(BaseModel):
    """Un modelo o un conjunto de demostraciones de la biblioteca.

    Si su carpeta no se puede leer, sale igualmente con ``valido: false`` y el motivo;
    entonces los campos que dependen de la ficha pueden faltar (null).
    """

    id: str
    entorno: TipoEntorno
    tipo: TipoElemento
    nombre: str
    origen: OrigenElemento | None
    creado: str | None
    descripcion: str | None
    fuente: Literal["experto", "humano", "combinado"] | None = Field(
        description="Solo en demostraciones: quién jugó las partidas.")
    artefacto: str | None = Field(description="Artefacto de W&B del que se descargó, si "
                                              "vino de allí (nombre:versión).")
    valido: bool
    motivo: str | None
    detalles: dict[str, Any] = Field(description="Lo principal de sus metadatos (pasos y "
                                                 "retorno del experto, partidas y pares "
                                                 "de unas demostraciones…).")


class DemostracionApi(BaseModel):
    """Un conjunto de demostraciones, como lo usa el formulario de entrenar."""

    id: str
    entorno: TipoEntorno
    nombre: str
    tipo: Literal["conjunto", "subconjunto"]
    partidas: Positivo | None
    pares: Positivo | None
    fuente: Literal["experto", "humano", "combinado"] | None
    origen: OrigenElemento | None
    valido: bool
    motivo: str | None
