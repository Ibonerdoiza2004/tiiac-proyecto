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

No registra en W&B: el artefacto de datos llegará con el módulo de seguimiento.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.data.demostraciones import generar_dataset  # noqa: E402
from src.training.trpo import (cargar_politica_experto,  # noqa: E402
                               comprobar_directorio_libre)
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402


def main() -> int:
    """Graba las demostraciones con el experto, crea los splits y devuelve el código de salida.

    Códigos: 0 si todo va bien; 1 si la huella de las partidas no coincide con
    --huella-esperada (los archivos se conservan para compararlos); 2 si la configuración, el
    experto o la carpeta de salida no son válidos, sin grabar nada.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="configuración base")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value, help="perfil de entorno")
    ap.add_argument("--experto", type=Path, required=True,
                    help="carpeta del experto (modelo.zip y experto.json)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta nueva para los datos "
                         "(por defecto data/demostraciones/<entorno>/<fecha>)")
    ap.add_argument("--huella-esperada", default=None,
                    help="huella de las partidas congeladas, para verificar la regeneración")
    args = ap.parse_args()

    try:
        cfg = resolver_configuracion(args.config, args.entorno)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("data") / "demostraciones" / args.entorno
                             / datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        comprobar_directorio_libre(salida)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2
    try:
        politica, metadatos = cargar_politica_experto(args.experto, cfg)
    except (OSError, ValueError) as error:
        print(f"error con el experto: {error}", file=sys.stderr)
        return 2

    try:
        resultado = generar_dataset(cfg, politica, salida, metadatos)
    except ValueError as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    retornos = resultado["retornos"]
    print(f"{len(retornos)} partidas y {resultado['pasos']} pasos guardados en {salida}")
    print(f"retorno medio: {sum(retornos) / len(retornos):.1f} "
          f"(mínimo {min(retornos):.1f}, máximo {max(retornos):.1f})")
    print(f"train: {resultado['train']}; validación: {resultado['validacion']}")
    print(f"huella de {resultado['demostraciones'].name}: {resultado['huella_dataset']}")
    print(f"huella de {resultado['splits'].name}: {resultado['huella_splits']}")
    print(f"huella de las partidas: {resultado['huella_partidas']}")
    codigo = metadatos["procedencia"]["codigo"]
    if codigo["commit"] is None:
        print("aviso: no se pudo leer el commit de git; los metadatos no dicen con qué código "
              "se grabaron", file=sys.stderr)
    elif codigo["cambios_sin_commit"]:
        print("aviso: había cambios sin commit; estos datos no se pueden regenerar "
              "exactamente desde el repositorio", file=sys.stderr)
    if args.huella_esperada is not None and args.huella_esperada != resultado["huella_partidas"]:
        print(f"error: la huella de las partidas no coincide con la esperada "
              f"({args.huella_esperada}); son distintas de las congeladas", file=sys.stderr)
        return 1
    if args.huella_esperada is not None:
        print("ok: las partidas coinciden con las congeladas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
