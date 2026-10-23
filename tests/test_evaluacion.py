"""Pruebas del protocolo común de predicción, evaluación y checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.models.factory import PoliticaBC
from src.training.evaluacion import (cargar_politica_bc, config_evaluacion,
                                     evaluar_final, evaluar_politica,
                                     guardar_politica_bc, predecir_accion)
from src.training.trpo import cargar_experto, entrenar_experto, guardar_experto
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import fijar_seed


def _cfg() -> dict:
    return {"entorno": {"id": "CartPole-v1"},
            "bc": {"activacion": "tanh", "capas_ocultas": [], "log_std_inicial": 0.0}}


def test_bc_discreta_respeta_start_y_modo_determinista():
    espacio = spaces.Discrete(3, start=-2)
    politica = PoliticaBC(2, 3, False, [], "tanh")
    with torch.no_grad():
        politica.red[-1].bias.copy_(torch.tensor([0.0, 3.0, 1.0]))
    accion = predecir_accion(politica, np.zeros(2, dtype=np.float32), espacio, True)
    assert accion == -1 and espacio.contains(accion)
    assert not politica.training


@pytest.mark.parametrize("semilla", [1, 5, 101])
def test_bc_estocastica_es_reproducible_y_no_altera_rng_global(semilla):
    espacio = spaces.Discrete(2)
    politica = PoliticaBC(2, 2, False, [], "tanh")
    antes = torch.random.get_rng_state()
    a = predecir_accion(politica, np.ones(2, dtype=np.float32), espacio, False, semilla)
    b = predecir_accion(politica, np.ones(2, dtype=np.float32), espacio, False, semilla)
    assert a == b and torch.equal(antes, torch.random.get_rng_state())


def test_acciones_box_de_bc_se_recortan_y_son_float32():
    espacio = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    politica = PoliticaBC(2, 2, True, [], "tanh", 0.0)
    with torch.no_grad():
        politica.red[-1].bias.copy_(torch.tensor([4.0, -4.0]))
    accion = predecir_accion(politica, np.zeros(2, dtype=np.float32), espacio, True)
    assert accion.dtype == np.float32
    assert np.array_equal(accion, np.array([1.0, -1.0], dtype=np.float32))
    assert espacio.contains(accion)


def test_predecir_rechaza_box_escalar_incompatible_sin_indexerror():
    politica = PoliticaBC(2, 2, True, [], "tanh", 0.0)
    espacio = spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32)
    with pytest.raises(ValueError, match="no coincide"):
        predecir_accion(politica, np.zeros(2, dtype=np.float32), espacio, True)


def test_aleatorio_y_evaluacion_se_reproducen_con_las_mismas_semillas():
    cfg = _cfg()
    a = evaluar_politica("aleatorio", cfg, 3, 42)
    b = evaluar_politica("aleatorio", cfg, 3, 42)
    assert a == b
    assert len(a["retornos"]) == len(a["longitudes"]) == 3
    assert a["retorno_medio"] == float(np.mean(a["retornos"]))


def test_evaluar_final_usa_configuracion_y_comparte_protocolo():
    cfg = {**_cfg(), "evaluacion": {"episodios": 3, "semilla": 42,
                                    "determinista": True}}
    assert config_evaluacion(cfg) == {"episodios": 3, "semilla": 42, "determinista": True}
    assert evaluar_final("aleatorio", cfg) == evaluar_politica("aleatorio", cfg, 3, 42, True)


@pytest.mark.parametrize("evaluacion", [
    None,
    {},
    {"episodios": 0, "semilla": 0, "determinista": True},
    {"episodios": True, "semilla": 0, "determinista": True},
    {"episodios": 1},
    {"episodios": 1, "semilla": 0},
    {"episodios": 1, "semilla": -1, "determinista": True},
    {"episodios": 1, "semilla": 0, "determinista": 1},
])
def test_config_evaluacion_rechaza_seccion_o_valores_invalidos(evaluacion):
    cfg = _cfg()
    if evaluacion is not None:
        cfg["evaluacion"] = evaluacion
    with pytest.raises(ValueError):
        config_evaluacion(cfg)


@pytest.mark.parametrize("episodios,semilla", [(0, 0), (-1, 0), (1.5, 0), (1, -2), (1, True)])
def test_evaluacion_rechaza_episodios_o_semilla_invalidos(episodios, semilla):
    with pytest.raises(ValueError):
        evaluar_politica("aleatorio", _cfg(), episodios, semilla)


def test_checkpoint_bc_conserva_pesos_buffers_y_metadatos(tmp_path: Path):
    politica = PoliticaBC(4, 2, False, [5, 3], "tanh")
    politica.ajustar_normalizacion(torch.arange(20, dtype=torch.float32).reshape(5, 4), 1e-6)
    politica.eval()
    datos = {"semilla": 17, "subconjunto": 4, "huella_dataset": "a" * 64}
    ruta = tmp_path / "bc.pt"
    cfg = _cfg()
    cfg["bc"]["capas_ocultas"] = [5, 3]
    guardar_politica_bc(ruta, politica, cfg, datos)
    cargada, metadatos = cargar_politica_bc(ruta)
    assert not cargada.training
    assert metadatos["entorno"] == "CartPole-v1"
    assert metadatos["huella_dataset"] == "a" * 64
    for clave, tensor in politica.state_dict().items():
        assert torch.equal(tensor, cargada.state_dict()[clave])
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(politica(x), cargada(x))


def test_checkpoint_bc_no_sobrescribe_y_rechaza_archivo_danado(tmp_path: Path):
    ruta = tmp_path / "bc.pt"
    politica = PoliticaBC(2, 2, False, [], "tanh")
    datos = {"semilla": 0, "subconjunto": 1, "huella_dataset": "b" * 64}
    guardar_politica_bc(ruta, politica, _cfg(), datos)
    with pytest.raises(FileExistsError):
        guardar_politica_bc(ruta, politica, _cfg(), datos)
    ruta.write_bytes(b"no es un checkpoint")
    with pytest.raises(ValueError):
        cargar_politica_bc(ruta)


def test_checkpoint_bc_rechaza_configuracion_que_no_reconstruye_la_red(tmp_path: Path):
    politica = PoliticaBC(2, 2, False, [4], "tanh")
    datos = {"semilla": 0, "subconjunto": 1, "huella_dataset": "d" * 64}
    with pytest.raises(ValueError, match="capas_ocultas"):
        guardar_politica_bc(tmp_path / "bc.pt", politica, _cfg(), datos)


def test_evaluacion_no_modifica_politica_bc_ni_crea_gradientes():
    politica = PoliticaBC(4, 2, False, [8], "tanh")
    estado = copy.deepcopy(politica.state_dict())
    evaluar_politica(politica, _cfg(), 2, 8)
    assert not politica.training
    assert all(param.grad is None for param in politica.parameters())
    assert all(torch.equal(valor, politica.state_dict()[clave])
               for clave, valor in estado.items())


def test_bc_estocastica_usa_un_flujo_aislado_por_episodio():
    cfg = {**_cfg(), "evaluacion": {"episodios": 3, "semilla": 42,
                                    "determinista": False}}
    politica = PoliticaBC(4, 2, False, [], "tanh")
    rng_antes = torch.random.get_rng_state()
    primero = evaluar_final(politica, cfg)
    segundo = evaluar_final(politica, cfg)
    assert primero == segundo
    assert torch.equal(rng_antes, torch.random.get_rng_state())


def test_experto_guardado_conserva_evaluacion_normalizada(tmp_path: Path):
    raiz = Path(__file__).resolve().parent.parent
    cfg = resolver_configuracion(raiz / "configs" / "default.yaml", "cartpole")
    cfg["entorno"]["id"] = "Pendulum-v1"
    cfg["trpo"].update(pasos_por_iteracion=64, n_entornos=1, batch_size_valor=32,
                       capas_ocultas=[8, 8], normalizar_observaciones=True)
    cfg["experto"].update(pasos_totales=128, retorno_objetivo=1_000_000,
                           evaluar_cada=2, episodios_evaluacion=2)
    fijar_seed(13)
    modelo, resumen = entrenar_experto(cfg)
    assert resumen["normalizacion"] is not None
    ruta = tmp_path / "experto"
    guardar_experto(ruta, modelo, cfg, resumen)
    cargado, metadatos = cargar_experto(ruta)
    assert evaluar_politica(
        cargado, cfg, 3, 7, True, metadatos["resumen"]["normalizacion"]
    ) == evaluar_politica(modelo, cfg, 3, 7, True, resumen["normalizacion"])



# --- regresiones de la auditoría del 22 de octubre ------------------------------------

_DATOS = {"semilla": 0, "subconjunto": 4, "huella_dataset": "c" * 64}


def _politica_con_salida(activacion: str) -> PoliticaBC:
    """Política discreta con la última capa no nula: la activación cambia la salida."""
    fijar_seed(0)
    politica = PoliticaBC(2, 2, False, [4], activacion)
    with torch.no_grad():
        politica.red[-1].weight.normal_()
    return politica.eval()


@pytest.mark.parametrize("activacion", ["tanh", "relu"])
def test_checkpoint_bc_guarda_la_activacion_de_la_red(tmp_path: Path, activacion):
    """Lo guardado sale de la propia red: al cargarla da las mismas salidas con cualquier
    activación de ACTIVACIONES."""
    from src.models.factory import ACTIVACIONES
    assert set(ACTIVACIONES) == {"tanh", "relu"}
    politica = _politica_con_salida(activacion)
    cfg = _cfg()
    cfg["bc"].update(activacion=activacion, capas_ocultas=[4])
    guardar_politica_bc(tmp_path / "bc.pt", politica, cfg, _DATOS)
    cargada, _ = cargar_politica_bc(tmp_path / "bc.pt")
    assert cargada.activacion == activacion
    x = torch.tensor([[-3.0, 2.0], [1.5, -0.5]])
    assert torch.equal(politica(x), cargada(x))


def test_checkpoint_bc_rechaza_una_activacion_distinta_de_la_red(tmp_path: Path):
    """Regresión: se guardaba la activación de la config (tanh) aunque la red fuera relu,
    y al cargarla cambiaban sus salidas."""
    cfg = _cfg()
    cfg["bc"].update(activacion="tanh", capas_ocultas=[4])
    with pytest.raises(ValueError, match="activacion.*'relu'.*'tanh'"):
        guardar_politica_bc(tmp_path / "bc.pt", _politica_con_salida("relu"), cfg, _DATOS)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error", [OSError("disco lleno"), KeyboardInterrupt()])
def test_checkpoint_bc_no_queda_a_medias_si_falla_al_escribir(tmp_path: Path, monkeypatch,
                                                               error):
    """Regresión: un torch.save fallido dejaba un archivo ilegible que además impedía
    volver a guardar (FileExistsError)."""
    import src.training.evaluacion as evaluacion
    original = evaluacion.torch.save

    def a_medias(objeto, ruta):
        Path(ruta).write_bytes(b"PK\x03\x04parcial")
        raise error

    monkeypatch.setattr(evaluacion.torch, "save", a_medias)
    with pytest.raises(type(error)):
        guardar_politica_bc(tmp_path / "bc.pt", PoliticaBC(2, 2, False, [], "tanh"), _cfg(),
                            _DATOS)
    assert list(tmp_path.iterdir()) == []  # ni el checkpoint ni el temporal
    monkeypatch.setattr(evaluacion.torch, "save", original)
    guardar_politica_bc(tmp_path / "bc.pt", PoliticaBC(2, 2, False, [], "tanh"), _cfg(), _DATOS)
    cargar_politica_bc(tmp_path / "bc.pt")


@pytest.mark.parametrize("cambio,mensaje", [
    ({"huella_dataset": "x"}, "SHA-256"),
    ({"huella_dataset": "A" * 64}, "SHA-256"),
    ({"huella_dataset": None}, "SHA-256"),
    ({"subconjunto": 0}, "subconjunto"),
    ({"subconjunto": -4}, "subconjunto"),
    ({"subconjunto": "cuatro"}, "subconjunto"),
    ({"subconjunto": True}, "subconjunto"),
    ({"subconjunto": 4.0}, "subconjunto"),
    ({"semilla": -1}, "semilla"),
])
def test_checkpoint_bc_valida_los_metadatos(tmp_path: Path, cambio, mensaje):
    """Regresión: la huella y el subconjunto se guardaban sin comprobar."""
    with pytest.raises(ValueError, match=mensaje):
        guardar_politica_bc(tmp_path / "bc.pt", PoliticaBC(2, 2, False, [], "tanh"), _cfg(),
                            {**_DATOS, **cambio})
    assert not list(tmp_path.iterdir())


def test_checkpoint_bc_admite_numpy_y_rechaza_nan_en_la_config(tmp_path: Path):
    """Regresión: un entero de NumPy en la configuración impedía guardar."""
    cfg = {**_cfg(), "experimento": {"seed": np.int64(3), "escala": np.float32(0.5)}}
    guardar_politica_bc(tmp_path / "bc.pt", PoliticaBC(2, 2, False, [], "tanh"), cfg,
                        {**_DATOS, "semilla": np.int64(7), "subconjunto": np.int64(4)})
    _, metadatos = cargar_politica_bc(tmp_path / "bc.pt")
    assert metadatos["configuracion"]["experimento"] == {"seed": 3, "escala": 0.5}
    assert (metadatos["semilla"], metadatos["subconjunto"]) == (7, 4)
    with pytest.raises(ValueError, match="JSON finito"):
        guardar_politica_bc(tmp_path / "nan.pt", PoliticaBC(2, 2, False, [], "tanh"),
                            {**_cfg(), "extra": float("nan")}, _DATOS)


@pytest.mark.parametrize("agente", ["aleatorio", "bc"])
def test_solo_trpo_admite_estadisticas(agente):
    """Regresión: BC y el agente aleatorio las aceptaban y las ignoraban sin avisar."""
    estadisticas = {"media": [100.0, 100.0], "varianza": [1.0, 1.0], "epsilon": 1e-8,
                    "limite": 10.0}
    politica = "aleatorio" if agente == "aleatorio" else PoliticaBC(2, 2, False, [], "tanh")
    with pytest.raises(ValueError, match="Solo TRPO"):
        predecir_accion(politica, np.zeros(2, np.float32), spaces.Discrete(2), True, None,
                        estadisticas)
    with pytest.raises(ValueError, match="Solo TRPO"):
        evaluar_politica(politica, _cfg(), 1, 0, True, estadisticas)


_ESTADISTICAS = {"media": [0.0, 1.0], "varianza": [4.0, 1.0], "epsilon": 1e-8, "limite": 10.0}


@pytest.mark.parametrize("cambio", [
    {"varianza": None}, {"media": [0.0]}, {"media": [[0.0, 1.0]]}, {"varianza": [1.0]},
    {"varianza": [-1.0, 1.0]}, {"media": [np.nan, 0.0]}, {"epsilon": 0.0}, {"epsilon": True},
    {"limite": -1.0}, {"limite": float("inf")},
])
def test_normalizar_observacion_rechaza_estadisticas_no_validas(cambio):
    """Regresión: un diccionario incompleto daba KeyError en vez de ValueError."""
    from src.training.evaluacion import normalizar_observacion
    estadisticas = {**_ESTADISTICAS, **cambio}
    if cambio == {"varianza": None}:
        del estadisticas["varianza"]
    with pytest.raises(ValueError):
        normalizar_observacion(np.zeros(2), estadisticas)


def test_normalizar_observacion_admite_una_observacion_o_un_lote():
    from src.training.evaluacion import normalizar_observacion
    observacion = np.array([2.0, 3.0])
    assert normalizar_observacion(observacion, None) is observacion
    esperada = np.array([1.0, 2.0], np.float32)
    np.testing.assert_allclose(normalizar_observacion(observacion, _ESTADISTICAS), esperada,
                               rtol=1e-6)
    lote = normalizar_observacion(np.stack([observacion, observacion]), _ESTADISTICAS)
    assert lote.shape == (2, 2) and lote.dtype == np.float32
    with pytest.raises(ValueError):
        normalizar_observacion(np.zeros(3), _ESTADISTICAS)


def test_bc_estocastica_juega_cada_episodio_con_su_semilla():
    """Regresión: con la semilla por paso en vez de por episodio pasaban todos los tests.
    Se compara con partidas jugadas a mano: cada una empieza con reiniciar_con_semilla y
    muestrea dentro de generador_torch_aislado(su semilla)."""
    from src.data.entornos import crear_entorno
    from src.utils.seed import (generador_torch_aislado, reiniciar_con_semilla,
                                semillas_independientes)
    fijar_seed(5)
    politica = PoliticaBC(4, 2, False, [8], "tanh")
    with torch.no_grad():
        politica.red[-1].weight.normal_(std=0.3)  # acciones inciertas: el muestreo cuenta
    resultado = evaluar_politica(politica, _cfg(), 4, 42, False)
    env = crear_entorno(_cfg(), None)
    retornos = []
    try:
        for semilla in semillas_independientes(42, 4):
            with generador_torch_aislado(semilla):
                observacion, _ = reiniciar_con_semilla(env, semilla)
                retorno, fin = 0.0, False
                while not fin:
                    accion = predecir_accion(politica, observacion, env.action_space, False)
                    observacion, recompensa, terminado, truncado, _ = env.step(accion)
                    retorno += float(recompensa)
                    fin = terminado or truncado
            retornos.append(retorno)
    finally:
        env.close()
    assert resultado["retornos"] == retornos
    # Cada episodio depende solo de su semilla, no de cuántos haya.
    assert evaluar_politica(politica, _cfg(), 2, 42, False)["retornos"] == retornos[:2]
