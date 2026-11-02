#!/usr/bin/env python3
"""Genera las demostraciones del experto y sus splits, y verifica su huella.

Los datos NO van al repositorio. No se descargan: se graban con el experto TRPO
propio, que tampoco se versiona (scripts/entrenar_experto.py). Lo que va es este
script, que es el procedimiento reproducible para obtenerlos:

    python scripts/entrenar_experto.py --entorno cartpole
    python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha>
    python scripts/get_data.py --entorno cartpole --experto <dir> --huella-esperada <sha256>

Escribe demostraciones.npz y splits.json en una carpeta nueva (por defecto
data/demostraciones/<entorno>/<fecha>) e imprime tres huellas SHA-256: la de
cada archivo y la de las partidas (huella_partidas, solo su contenido). Los
splits citan la del NPZ, que también depende de los metadatos (commit,
versiones). --huella-esperada compara la de las partidas, que con el mismo
experto, configuración y máquina no cambia aunque cambie el commit. En otra
máquina los números del experto pueden variar en los últimos decimales y la
huella no está garantizada.

Registra el run de tipo datos y sube el NPZ y sus splits como artefacto. Usa
--grupo para reunirlo con los runs del experto, BC, GAIL y comparación; usa
--sin-wandb para desactivar el seguimiento.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.training.flujo import generar_y_registrar_datos  # noqa: E402
from src.utils.archivos import marca_de_tiempo  # noqa: E402
from src.utils.configuracion import crear_parser, resolver_configuracion  # noqa: E402
from src.utils.tracking import comprobar_wandb, crear_grupo  # noqa: E402


def main() -> int:
    """Graba las demostraciones con el experto, crea los splits y devuelve el código de salida.

    Códigos: 0 si todo va bien; 1 si la huella de las partidas no coincide con
    --huella-esperada (los archivos se conservan para compararlos); 2 si la configuración, el
    experto o la carpeta de salida no son válidos, sin grabar nada.

    Returns:
        Código de salida 0, 1 o 2 según el resultado de la etapa.

    Raises:
        Ninguna: los errores esperados se presentan al usuario como códigos de salida.
    """
    ap = crear_parser(__doc__)
    ap.add_argument("--experto", type=Path, required=True,
                    help="carpeta del experto (modelo.zip y experto.json)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta nueva para los datos "
                         "(por defecto data/demostraciones/<entorno>/<fecha>)")
    ap.add_argument("--huella-esperada", default=None,
                    help="huella de las partidas congeladas, para verificar la regeneración")
    args = ap.parse_args()

    try:
        ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
    except ValueError as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or Path("data") / "demostraciones" / args.entorno / marca_de_tiempo()
    try:
        comprobar_wandb(cfg, exigir_sesion=True)
    except RuntimeError as error:
        print(f"error de W&B: {error}", file=sys.stderr)
        return 2

    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    try:
        resultado = generar_y_registrar_datos(cfg, args.experto, salida, grupo,
                                              args.huella_esperada)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"error con el experto: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    if not resultado["coincide"]:
        print(f"huella de demostraciones.npz: {resultado['huella_dataset']}")
        print(f"huella de splits.json: {resultado['huella_splits']}")
        print(f"huella de las partidas: {resultado['huella_partidas']}")
        print(f"error: la huella de las partidas no coincide con la esperada "
              f"({args.huella_esperada}); son distintas de las congeladas", file=sys.stderr)
        return 1
    _imprimir(resultado, salida)
    if args.huella_esperada is not None:
        print("ok: las partidas coinciden con las congeladas")
    return 0


def _imprimir(resultado: dict, salida: Path) -> None:
    """Imprime el resumen de las partidas grabadas, los splits y sus huellas."""
    retornos = resultado["retornos"]
    print(f"{len(retornos)} partidas y {resultado['pasos']} pasos guardados en {salida}")
    print(f"retorno medio: {sum(retornos) / len(retornos):.1f} "
          f"(mínimo {min(retornos):.1f}, máximo {max(retornos):.1f})")
    print(f"train: {resultado['train']}; validación: {resultado['validacion']}")
    print(f"huella de {resultado['demostraciones'].name}: {resultado['huella_dataset']}")
    print(f"huella de {resultado['splits'].name}: {resultado['huella_splits']}")
    print(f"huella de las partidas: {resultado['huella_partidas']}")


if __name__ == "__main__":
    raise SystemExit(main())
