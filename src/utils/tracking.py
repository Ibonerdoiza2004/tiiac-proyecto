"""Seguimiento de experimentos con Weights & Biases.

Este módulo mantiene la dependencia de W&B fuera del resto del proyecto y
permite ejecutar tests y smoke sin importar el paquete.
"""

from __future__ import annotations

import copy
import sys
import time
import uuid
from numbers import Real
from pathlib import Path
from typing import Any

_ULTIMOS_PASOS: dict[int, tuple[Any, int]] = {}

# Caracteres de la huella SHA-256 que lleva el nombre de un artefacto; los nombres
# congelados de configs/ (datos.artefacto_*) siguen esta longitud.
CARACTERES_HUELLA_ARTEFACTO = 12
# Segundos que espera wandb.login antes de rendirse, para no bloquear el arranque sin red.
ESPERA_SESION = 10


def crear_grupo() -> str:
    """Crea un identificador de 12 caracteres hexadecimales para agrupar una ejecución."""
    return uuid.uuid4().hex[:12]


def _wandb_activo(cfg: dict) -> bool:
    """Indica si la configuración habilita cualquier modo de W&B."""
    return cfg.get("wandb", {}).get("modo", "online") != "disabled"


def comprobar_wandb(cfg: dict, exigir_sesion: bool = False) -> Any | None:
    """Carga W&B si está activo y, cuando se pide, valida la sesión sin prompt.

    Args:
        cfg: Configuración resuelta.
        exigir_sesion: Valida credenciales en modo online antes de entrenar.

    Returns:
        El módulo W&B si está activo; ``None`` en modo disabled.

    Raises:
        RuntimeError: Si falta W&B o no se puede validar una sesión online.
    """
    if not _wandb_activo(cfg):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb está activo, pero el paquete wandb no está instalado") from error
    if exigir_sesion and cfg.get("wandb", {}).get("modo", "online") == "online":
        try:
            sesion = wandb.login(anonymous="never", timeout=ESPERA_SESION, verify=True)
        except Exception as error:  # noqa: BLE001 - W&B envuelve errores de red y credenciales
            raise RuntimeError(f"no se pudo validar la sesión de W&B: {error}") from error
        if not sesion:
            raise RuntimeError("no hay una sesión válida de W&B; ejecuta wandb login")
    return wandb


def abrir_run(cfg: dict, grupo: str | None = None, job_type: str | None = None,
              nombre: str | None = None, etiquetas: list[str] | None = None) -> Any | None:
    """Abre un run con su configuración y metadatos; devuelve None si está disabled.

    Args:
        cfg: Configuración resuelta del experimento.
        grupo: Identificador compartido por los runs de una ejecución completa.
        job_type: Tipo de trabajo (experto, datos, bc, gail o comparacion).
        nombre: Nombre legible opcional del run.
        etiquetas: Etiquetas para filtrar entorno, método, cantidad y repetición.
    """
    wcfg = cfg.get("wandb", {})
    wandb = comprobar_wandb(cfg)
    if wandb is None:
        return None
    return wandb.init(
        project=wcfg.get("proyecto", "tiiac-2026"),
        entity=wcfg.get("entidad") or None,
        name=nombre or cfg["experimento"]["nombre"],
        config=cfg,
        mode=wcfg.get("modo", "online"),
        group=grupo,
        job_type=job_type,
        tags=list(etiquetas or []),
    )


def nombre_run(cfg: dict, metodo: str, cantidad: int | None = None,
               repeticion: int | None = None) -> str:
    """Nombre legible: <experimento>-<metodo>-<entorno>[-c<cantidad>-r<repeticion>]."""
    entorno = cfg.get("origen", {}).get("entorno", cfg["entorno"]["id"])
    nombre = f"{cfg['experimento']['nombre']}-{metodo}-{entorno}"
    if cantidad is not None:
        nombre += f"-c{cantidad}-r{repeticion}"
    return nombre


def abrir_run_barrido(cfg: dict, metodo: str, cantidad: int, repeticion: int,
                      grupo: str | None = None) -> Any | None:
    """Abre un run de BC o GAIL con grupo, job_type y etiquetas identificativas."""
    entorno = cfg.get("origen", {}).get("entorno", cfg["entorno"]["id"])
    cfg_run = copy.deepcopy(cfg)
    cfg_run["experimento"]["nombre"] = nombre_run(cfg, metodo, cantidad, repeticion)
    return abrir_run(cfg_run, grupo=grupo, job_type=metodo,
                     etiquetas=[f"entorno:{entorno}", f"metodo:{metodo}",
                                f"cantidad:{cantidad}", f"repeticion:{repeticion}"])


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra valores escalares si hay un run, sin importar W&B en disabled."""
    if run is None:
        return
    escalares = {clave: valor for clave, valor in metricas.items()
                 if isinstance(valor, Real)}
    if escalares:
        if paso is not None:
            registro = _ULTIMOS_PASOS.get(id(run))
            anterior = registro[1] if registro is not None and registro[0] is run else None
            if anterior is not None and paso < anterior:
                print(f"aviso: se omiten métricas con paso W&B decreciente: "
                      f"{paso} < {anterior}", file=sys.stderr)
                return
            _ULTIMOS_PASOS[id(run)] = (run, paso)
        run.log(escalares, step=paso)


def registrar_resumen(run: Any | None, resumen: dict[str, Any]) -> None:
    """Añade al summary del run la procedencia y el resumen final del entrenamiento."""
    summary = getattr(run, "summary", None) if run is not None else None
    if summary is not None:
        from src.utils.procedencia import estado_git, versiones_y_hardware

        summary.update({**resumen, "codigo": estado_git(),
                        "entorno_ejecucion": versiones_y_hardware()})


def nombre_artefacto(tipo: str, huella: str, entorno: str | None = None,
                     cantidad: int | None = None, repeticion: int | None = None) -> str:
    """Nombre de un artefacto: <tipo>[-<entorno>][-c<cantidad>-r<repeticion>]-<huella corta>.

    Args:
        tipo: experto, demostraciones, bc o gail.
        huella: SHA-256 del archivo que identifica el artefacto.
        entorno: Id del entorno; las demostraciones no lo llevan.
        cantidad: Subconjunto de BC y GAIL; va siempre con repeticion.
        repeticion: Repetición de BC y GAIL; va siempre con cantidad.

    Returns:
        El nombre sin versión, con los primeros CARACTERES_HUELLA_ARTEFACTO de la huella.

    Raises:
        ValueError: Si la huella no es un SHA-256 o solo se da cantidad o repetición.
    """
    from src.data.splits import comprobar_huella

    comprobar_huella(huella)
    if (cantidad is None) != (repeticion is None):
        raise ValueError("cantidad y repeticion van juntas en el nombre de un artefacto")
    partes = [tipo] + ([entorno] if entorno is not None else [])
    if cantidad is not None:
        partes += [f"c{cantidad}", f"r{repeticion}"]
    return "-".join([*partes, huella[:CARACTERES_HUELLA_ARTEFACTO]])


def subir_artefacto(run: Any | None, nombre: str, tipo: str,
                    archivos: list[Path], metadatos: dict[str, Any] | None = None,
                    aliases: list[str] | None = None) -> Any | None:
    """Sube archivos locales como un artefacto versionado relacionado con el run.

    Los caminos se conservan como nombres de archivo dentro del artefacto, sin
    subir la carpeta global wandb/ ni registros locales.
    """
    if run is None or not hasattr(run, "log_artifact"):
        return None
    wandb = comprobar_wandb({"wandb": {"modo": "offline"}})
    artefacto = wandb.Artifact(nombre, type=tipo, metadata=metadatos or {})
    for archivo in archivos:
        artefacto.add_file(str(Path(archivo)), name=Path(archivo).name)
    return run.log_artifact(artefacto, aliases=aliases or ["latest"])


def usar_artefacto(run: Any | None, nombre: str,
                   tipo: str = "demostraciones") -> Any | None:
    """Declara un artefacto de entrada si existe, sin bloquear el entrenamiento si falla."""
    if run is None or not hasattr(run, "use_artifact"):
        return None
    wandb = comprobar_wandb({"wandb": {"modo": "offline"}})
    try:
        return run.use_artifact(nombre, type=tipo)
    except ValueError as error:
        print(f"aviso: no se encontró el artefacto {nombre}; se sigue sin registrar el linaje "
              f"({error})", file=sys.stderr)
        return None
    except wandb.errors.CommError as error:
        print(f"aviso: no se pudo consultar el artefacto {nombre}; "
              f"se sigue sin registrar el linaje ({error})", file=sys.stderr)
        return None


def descargar_artefacto(cfg: dict, nombre: str, destino: Path) -> Path:
    """Descarga la última versión de un artefacto del proyecto de W&B en destino.

    Usa la entidad, el proyecto y la sesión de W&B del usuario aunque
    wandb.modo sea disabled; no abre ningún run.

    Args:
        cfg: Configuración con la sección wandb.
        nombre: Nombre del artefacto, sin versión.
        destino: Carpeta donde se descargan sus archivos.

    Returns:
        La carpeta con los archivos descargados.

    Raises:
        RuntimeError: Si falta W&B, no hay sesión o no existe el artefacto.
    """
    wandb = comprobar_wandb({"wandb": {"modo": "offline"}})
    ruta = f"{cfg['wandb']['entidad']}/{cfg['wandb']['proyecto']}/{nombre}:latest"
    try:
        return Path(wandb.Api().artifact(ruta).download(root=str(destino)))
    except (ValueError, wandb.errors.Error) as error:
        raise RuntimeError(f"no se pudo descargar {ruta} (¿has iniciado sesión con "
                           f"wandb login?): {error}") from error


def terminar_run(run: Any | None, error: BaseException | None = None) -> None:
    """Cierra el run con código 0, 1 por fallo o 130 por Ctrl+C."""
    if run is not None:
        run.finish(exit_code=(0 if error is None else
                              130 if isinstance(error, KeyboardInterrupt) else 1))
        _ULTIMOS_PASOS.pop(id(run), None)


class RegistroProgreso:
    """Reenvía logs a W&B e imprime una línea por validación de GAIL."""

    def __init__(self, run: Any | None, etiqueta: str, pasos_por_iteracion: int,
                 iteraciones: int) -> None:
        """Guarda el run y el presupuesto necesarios para estimar el tiempo restante."""
        self.run, self.etiqueta = run, etiqueta
        self.pasos_por_iteracion, self.iteraciones = pasos_por_iteracion, iteraciones
        self.inicio = time.perf_counter()
        self.ultimas: dict[str, float] = {}

    def log(self, metricas: dict, step: int | None = None) -> None:
        """Reenvía métricas y, al validar, muestra retorno y tiempo estimado."""
        registrar(self.run, metricas, step)
        if "validacion/retorno_medio" not in metricas:
            self.ultimas = metricas
            return
        iteracion = int(step) // self.pasos_por_iteracion
        segundos = time.perf_counter() - self.inicio
        restantes = segundos / iteracion * (self.iteraciones - iteracion)
        print(f"{self.etiqueta} · iteración {iteracion}/{self.iteraciones} · {step} pasos · "
              f"validación {metricas['validacion/retorno_medio']:.1f} · D acierta "
              f"{self.ultimas.get('discriminador/acierto_agente', float('nan')):.2f} (agente) "
              f"{self.ultimas.get('discriminador/acierto_experto', float('nan')):.2f} (experto) · "
              f"{segundos:.0f} s · quedan ~{restantes / 60:.0f} min", flush=True)
