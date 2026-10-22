"""Seguimiento de experimentos con Weights & Biases.

Aislado en un modulo para que el resto del codigo no dependa de wandb y para
que los tests puedan correr con modo='disabled'.
"""

from __future__ import annotations

import time
from typing import Any


def abrir_run(cfg: dict) -> Any | None:
    """Abre un run de W&B, o devuelve None si esta desactivado."""
    wcfg = cfg.get("wandb", {})
    if wcfg.get("modo") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("aviso: wandb no esta instalado; se sigue sin registrar")
        return None

    return wandb.init(
        project=wcfg.get("proyecto", "tiiac-2026"),
        entity=wcfg.get("entidad") or None,
        name=cfg["experimento"]["nombre"],
        config=cfg,
        mode=wcfg.get("modo", "online"),
    )


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra metricas si hay run abierto. Silencioso si no."""
    if run is not None:
        run.log(metricas, step=paso)


class RegistroProgreso:
    """Hace de run de W&B para un entrenamiento por iteraciones e imprime su progreso.

    Reenvía cada llamada a log al run real, si lo hay, e imprime una línea
    cuando las medidas son de validación (validacion/retorno_medio): la
    iteración, los pasos, el retorno, los aciertos del discriminador de la
    última iteración y el tiempo que queda, estimado con el ritmo medido.
    """

    def __init__(self, run: Any | None, etiqueta: str, pasos_por_iteracion: int,
                 iteraciones: int) -> None:
        """Guarda el run real (o None) y lo necesario para calcular la iteración y el tiempo."""
        self.run, self.etiqueta = run, etiqueta
        self.pasos_por_iteracion, self.iteraciones = pasos_por_iteracion, iteraciones
        self.inicio = time.perf_counter()
        self.ultimas: dict[str, float] = {}

    def log(self, metricas: dict, step: int | None = None) -> None:
        """Reenvía las medidas e imprime una línea si son de validación."""
        if self.run is not None:
            self.run.log(metricas, step=step)
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
