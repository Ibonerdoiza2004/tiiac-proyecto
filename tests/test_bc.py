"""Pruebas de Behavioral Cloning: política, dataset, loaders y entrenamiento.

Se cubren acciones discretas (Discrete con y sin start) y continuas (Box de una
dimensión). Las partidas son inventadas, como prevé el reparto: no hace falta
experto ni entorno. Los entornos reales solo se usan para comprobar que la
política se construye con sus espacios.
"""

import copy
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from src.data.loaders import DatasetTransiciones, construir_loaders_bc, submuestrear_pares
from src.data.splits import crear_splits
from src.data.tipos import LoteTransiciones, Trayectoria
from src.models.factory import ACTIVACIONES, PoliticaBC, construir_politica_bc
from src.training.loop import entrenar_bc, entrenar_bc_subconjunto, evaluar_bc
from src.utils.seed import crear_generador, fijar_seed

RAIZ = Path(__file__).resolve().parent.parent
HUELLA = "a" * 64
OBSERVACION = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
DISCRETO = spaces.Discrete(2)
CONTINUO = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
CFG = {"bc": {
    "capas_ocultas": [100, 100], "activacion": "tanh", "log_std_inicial": 0.0,
    "frecuencia_pares": 1, "batch_size": 8, "optimizador": "adam",
    "learning_rate": 0.01, "epocas_maximas": 50, "paciencia": 5,
    "normalizar_observaciones": True, "epsilon_normalizacion": 1e-6,
}}
TIPOS = pytest.mark.parametrize("continua", [False, True], ids=["discreta", "continua"])


def _cfg(**cambios) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["bc"].update(cambios)
    return cfg


def _politica(continua: bool, capas=(16,), dim: int = 4, n: int = 2) -> PoliticaBC:
    return PoliticaBC(dim, n, continua, list(capas), "tanh", 0.0 if continua else None)


def _criterio(continua: bool) -> str:
    return "error_cuadratico" if continua else "perdida"


def _partida(identificador: int, pasos: int = 6, continua: bool = False,
             inicio_accion: int = 0) -> Trayectoria:
    """Partida exacta: estado t = 100 * id + t; acción discreta t % 2 + inicio o continua (t, -t)."""
    base = 100 * identificador + np.arange(pasos + 1, dtype=np.float32)
    observaciones = np.repeat(base[:, None], 4, axis=1)
    t = np.arange(pasos)
    acciones = (np.stack([t, -t], axis=1).astype(np.float32) if continua
                else (t % 2 + inicio_accion).astype(np.int64))
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    return Trayectoria(identificador, observaciones[:-1], acciones, observaciones[1:],
                       np.ones(pasos, dtype=np.float32), terminados, np.zeros(pasos, dtype=bool))


def _quince(continua: bool = False) -> list[Trayectoria]:
    """15 partidas de longitudes distintas (5 a 9 pasos) con IDs 0 a 14."""
    return [_partida(i, pasos=5 + i % 5, continua=continua) for i in range(15)]


def _splits(trayectorias):
    return crear_splits(trayectorias, 10, 5, (1, 4, 7, 10), crear_generador(0), HUELLA)


def _lote(estados, acciones, ids=None) -> LoteTransiciones:
    estados, acciones = np.asarray(estados), np.asarray(acciones)
    n = len(estados)
    ids = np.zeros(n, dtype=np.int64) if ids is None else np.asarray(ids)
    return LoteTransiciones(estados, acciones, ids, np.arange(n, dtype=np.int64))


def _datos_aprendibles(continua: bool, n: int = 256, semilla: int = 0):
    """Pares en los que la acción depende del estado: signo de x0 o combinación lineal."""
    estados = np.random.default_rng(semilla).normal(size=(n, 4)).astype(np.float32)
    if continua:
        acciones = np.stack([estados[:, 0] - estados[:, 1], 0.5 * estados[:, 2]], axis=1)
        return estados, acciones.astype(np.float32)
    return estados, (estados[:, 0] > 0).astype(np.int64)


def _loader(estados, acciones, espacio, batch_size: int = 32, barajar: bool = False):
    return DataLoader(DatasetTransiciones(_lote(estados, acciones), espacio),
                      batch_size=batch_size, shuffle=barajar)


class RunFalso:
    """Imita un run de W&B guardando cada llamada a log."""

    def __init__(self):
        self.registros: list[tuple[dict, int]] = []

    def log(self, metricas, step=None):
        self.registros.append((dict(metricas), step))

    def valores(self, clave):
        return [m[clave] for m, _ in self.registros if clave in m]


# --- argumentos de PoliticaBC -------------------------------------------------

@pytest.mark.parametrize("valor", [1, 4, np.int64(3), np.int32(1)])
def test_dimensiones_admiten_enteros_positivos(valor):
    politica = PoliticaBC(valor, valor, False, [2], "tanh")
    assert (politica.dim_observacion, politica.dim_accion) == (int(valor), int(valor))
    assert type(politica.dim_observacion) is int and type(politica.dim_accion) is int


@pytest.mark.parametrize("valor", [0, -1, 2.0, 2.5, True, False, np.bool_(True), "4", None])
@pytest.mark.parametrize("argumento", ["dim_observacion", "dim_accion"])
def test_dimensiones_rechazan_valores_no_validos(argumento, valor):
    argumentos = {"dim_observacion": 4, "dim_accion": 2, "continua": False,
                  "capas_ocultas": [8], "activacion": "tanh"}
    argumentos[argumento] = valor
    with pytest.raises(ValueError, match=argumento):
        PoliticaBC(**argumentos)


@pytest.mark.parametrize("valor", [0, 1, "si", None, 1.0])
def test_continua_debe_ser_booleano(valor):
    with pytest.raises(ValueError, match="continua"):
        PoliticaBC(4, 2, valor, [8], "tanh")


@pytest.mark.parametrize("capas", [[], [1], (100, 100), [np.int64(5), 3]])
def test_capas_ocultas_admite_secuencias_de_enteros_positivos(capas):
    assert isinstance(PoliticaBC(4, 2, False, capas, "tanh"), PoliticaBC)


@pytest.mark.parametrize("capas", [
    [0], [100, -1], [1.5], [True], [100, None], "100", b"1", 100, None, {100, 50},
])
def test_capas_ocultas_rechaza_valores_no_validos(capas):
    with pytest.raises(ValueError, match="capas_ocultas"):
        PoliticaBC(4, 2, False, capas, "tanh")


@pytest.mark.parametrize("activacion", ["Tanh", "sigmoid", "", None, 1])
def test_activacion_rechaza_valores_desconocidos(activacion):
    with pytest.raises(ValueError, match="activacion"):
        PoliticaBC(4, 2, False, [8], activacion)


def test_las_activaciones_admitidas_son_las_documentadas():
    assert set(ACTIVACIONES) == {"tanh", "relu"}


@pytest.mark.parametrize("valor", [0.0, -0.5, 1, np.float32(-1.0), np.int64(0)])
def test_log_std_inicial_fija_el_parametro_en_continuas(valor):
    politica = PoliticaBC(4, 3, True, [8], "tanh", valor)
    assert politica.log_std.shape == (3,) and politica.log_std.requires_grad
    assert torch.equal(politica.log_std.detach(), torch.full((3,), float(valor)))


@pytest.mark.parametrize("valor", [None, math.nan, math.inf, -math.inf, True, "0", [0.0]])
def test_log_std_inicial_es_obligatorio_y_finito_en_continuas(valor):
    with pytest.raises(ValueError, match="log_std_inicial"):
        PoliticaBC(4, 2, True, [8], "tanh", valor)


def test_log_std_inicial_no_se_admite_en_discretas():
    with pytest.raises(ValueError, match="log_std_inicial"):
        PoliticaBC(4, 2, False, [8], "tanh", 0.0)
    assert PoliticaBC(4, 2, False, [8], "tanh").log_std is None


# --- normalización de observaciones ---------------------------------------------

def test_normalizacion_empieza_como_identidad_y_no_se_entrena():
    politica = _politica(False)
    assert torch.equal(politica.media, torch.zeros(4)) and torch.equal(politica.escala, torch.ones(4))
    nombres = dict(politica.named_parameters())
    assert "media" not in nombres and "escala" not in nombres
    assert {"media", "escala"} <= politica.state_dict().keys()


@pytest.mark.parametrize("n", [2, 7, 100])
def test_ajustar_normalizacion_usa_media_y_desviacion_poblacional(n):
    estados = np.random.default_rng(n).normal([5.0, -2.0, 0.0, 300.0], [1.0, 0.01, 3.0, 50.0],
                                              size=(n, 4)).astype(np.float32)
    politica = _politica(False)
    politica.ajustar_normalizacion(torch.tensor(estados), 1e-6)
    datos = estados.astype(np.float64)
    assert np.allclose(politica.media.numpy(), datos.mean(axis=0), rtol=1e-6)
    assert np.allclose(politica.escala.numpy(), datos.std(axis=0, ddof=0), rtol=1e-6)
    assert politica.media.dtype == politica.escala.dtype == torch.float32


def test_la_politica_aplica_la_normalizacion_antes_de_la_red():
    politica = _politica(True)
    with torch.no_grad():
        politica.red[-1].weight.normal_()
    politica.ajustar_normalizacion(torch.randn(20, 4) * 10 + 3, 1e-6)
    x = torch.randn(5, 4)
    esperado = politica.red((x - politica.media) / politica.escala)
    assert torch.allclose(politica(x), esperado)


@pytest.mark.parametrize("desviacion,escala", [(0.0, 1.0), (1e-6, 1.0), (1e-3, 1e-3)])
def test_las_componentes_casi_constantes_solo_se_centran(desviacion, escala):
    """Desviación <= epsilon: escala 1. Justo por encima: se divide por la desviación."""
    columna = torch.tensor([7.0 - desviacion, 7.0 + desviacion], dtype=torch.float64)
    estados = torch.stack([columna, columna, columna, columna], dim=1).float()
    politica = _politica(False)
    politica.ajustar_normalizacion(estados, 1e-6)
    assert torch.allclose(politica.media, torch.full((4,), 7.0))
    assert torch.allclose(politica.escala, torch.full((4,), escala), rtol=1e-3)


@pytest.mark.parametrize("epsilon,escala", [(0.5, 1.0), (0.4999, 0.5)])
def test_el_limite_de_epsilon_es_inclusivo(epsilon, escala):
    """Columna [-0.5, 0.5]: desviación exactamente 0.5, representable sin error."""
    estados = torch.tensor([[-0.5] * 4, [0.5] * 4], dtype=torch.float64)
    politica = _politica(False)
    politica.ajustar_normalizacion(estados, epsilon)
    assert torch.equal(politica.escala, torch.full((4,), escala))


def test_con_un_solo_estado_solo_se_centra():
    politica = _politica(False)
    politica.ajustar_normalizacion(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 1e-6)
    assert torch.equal(politica.media, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(politica.escala, torch.ones(4))


def test_la_normalizacion_se_guarda_y_se_carga_con_el_state_dict():
    politica = _politica(True)
    politica.ajustar_normalizacion(torch.randn(30, 4) * 5 + 2, 1e-6)
    copia = _politica(True)
    copia.load_state_dict(politica.state_dict())
    x = torch.randn(6, 4)
    assert torch.equal(copia.media, politica.media) and torch.equal(copia.escala, politica.escala)
    assert torch.equal(copia(x), politica(x))


@pytest.mark.parametrize("estados", [
    torch.zeros(0, 4), torch.zeros(4), torch.zeros(3, 5), torch.zeros(3, 4, dtype=torch.int64),
    torch.tensor([[0.0, 0.0, math.nan, 0.0]]), torch.tensor([[0.0, math.inf, 0.0, 0.0]]),
    np.zeros((3, 4), dtype=np.float32),
])
def test_ajustar_normalizacion_rechaza_estados_no_validos(estados):
    with pytest.raises(ValueError, match="estados"):
        _politica(False).ajustar_normalizacion(estados, 1e-6)


@pytest.mark.parametrize("epsilon", [0, -1e-6, math.nan, math.inf, True, None, "1e-6"])
def test_ajustar_normalizacion_rechaza_epsilon_no_valido(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        _politica(False).ajustar_normalizacion(torch.randn(3, 4), epsilon)


# --- log_verosimilitud ------------------------------------------------------------

def test_log_verosimilitud_discreta_es_la_entropia_cruzada_cambiada_de_signo():
    politica = _politica(False, n=3)
    estados, acciones = torch.randn(10, 4), torch.randint(0, 3, (10,))
    esperada = -torch.nn.functional.cross_entropy(politica(estados), acciones, reduction="none")
    assert torch.allclose(politica.log_verosimilitud(estados, acciones), esperada, atol=1e-6)


@pytest.mark.parametrize("log_std", [0.0, -1.0, 0.7])
def test_log_verosimilitud_continua_es_la_de_la_gaussiana_diagonal(log_std):
    politica = PoliticaBC(4, 3, True, [8], "tanh", log_std)
    estados, acciones = torch.randn(10, 4), torch.randn(10, 3)
    media, sigma = politica(estados), math.exp(log_std)
    esperada = (-0.5 * ((acciones - media) / sigma) ** 2 - log_std
                - 0.5 * math.log(2 * math.pi)).sum(dim=1)
    obtenida = politica.log_verosimilitud(estados, acciones)
    assert torch.allclose(obtenida, esperada, atol=1e-5)
    assert torch.allclose(obtenida, politica.distribucion(estados).log_prob(acciones), atol=1e-6)


@pytest.mark.parametrize("acciones", [
    torch.tensor([0, 1, 2]), torch.tensor([0, -1, 1]), torch.tensor([0.0, 1.0, 1.0]),
    torch.tensor([[0], [1], [1]]), torch.tensor([0, 1]), torch.tensor([True, False, True]),
])
def test_log_verosimilitud_discreta_rechaza_acciones_no_validas(acciones):
    with pytest.raises(ValueError, match="acciones"):
        _politica(False).log_verosimilitud(torch.randn(3, 4), acciones)


@pytest.mark.parametrize("acciones", [
    torch.randn(3), torch.randn(3, 3), torch.randn(2, 2), torch.ones(3, 2, dtype=torch.int64),
])
def test_log_verosimilitud_continua_rechaza_acciones_no_validas(acciones):
    with pytest.raises(ValueError, match="acciones"):
        _politica(True).log_verosimilitud(torch.randn(3, 4), acciones)


# --- construir_politica_bc ------------------------------------------------------

@pytest.mark.parametrize("entorno", [
    "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1", "MountainCarContinuous-v0",
])
def test_construye_la_politica_con_los_espacios_reales(entorno):
    env = gym.make(entorno)
    try:
        politica = construir_politica_bc(CFG, env.observation_space, env.action_space)
        estados = torch.as_tensor(
            np.stack([env.observation_space.sample() for _ in range(3)]), dtype=torch.float32)
        continua = isinstance(env.action_space, spaces.Box)
        dim_accion = env.action_space.shape[0] if continua else env.action_space.n
    finally:
        env.close()
    assert politica.continua == continua
    assert politica.dim_observacion == env.observation_space.shape[0]
    assert politica.dim_accion == dim_accion
    assert politica(estados).shape == (3, dim_accion)


def test_discrete_con_start_da_n_salidas():
    politica = construir_politica_bc(CFG, OBSERVACION, spaces.Discrete(3, start=-1))
    assert (politica.dim_accion, politica.continua) == (3, False)


def test_box_de_varias_acciones_usa_log_std_de_la_config():
    cfg = _cfg(log_std_inicial=-0.5)
    politica = construir_politica_bc(cfg, OBSERVACION, spaces.Box(-1, 1, (6,), np.float32))
    assert (politica.dim_accion, politica.continua) == (6, True)
    assert torch.equal(politica.log_std.detach(), torch.full((6,), -0.5))


@pytest.mark.parametrize("ruta", ["configs/default.yaml", "configs/smoke.yaml"])
@pytest.mark.parametrize("espacio_accion", [DISCRETO, CONTINUO], ids=["discreta", "continua"])
def test_las_configs_del_proyecto_construyen_la_politica_del_paper(ruta, espacio_accion):
    cfg = yaml.safe_load((RAIZ / ruta).read_text(encoding="utf-8"))
    politica = construir_politica_bc(cfg, OBSERVACION, espacio_accion)
    lineales = [m for m in politica.red if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in lineales] == [100, 100, 2]
    assert sum(isinstance(m, torch.nn.Tanh) for m in politica.red) == 2


@pytest.mark.parametrize("espacio_accion", [
    spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
    spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
    spaces.MultiDiscrete([2, 3]),
    spaces.MultiBinary(2),
])
def test_rechaza_espacios_de_accion_no_compatibles(espacio_accion):
    with pytest.raises(ValueError, match="acción"):
        construir_politica_bc(CFG, OBSERVACION, espacio_accion)


@pytest.mark.parametrize("espacio_observacion", [
    spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
    spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
    spaces.Discrete(4),
])
def test_rechaza_observaciones_no_compatibles(espacio_observacion):
    with pytest.raises(ValueError, match="observación"):
        construir_politica_bc(CFG, espacio_observacion, DISCRETO)


@pytest.mark.parametrize("cfg", [
    {}, {"bc": None}, {"bc": {}}, {"bc": {"capas_ocultas": [8]}},
    {"bc": {"activacion": "tanh"}}, None, [],
])
def test_rechaza_configuraciones_sin_seccion_bc_completa(cfg):
    with pytest.raises(ValueError, match="bc.capas_ocultas"):
        construir_politica_bc(cfg, OBSERVACION, DISCRETO)


def test_log_std_inicial_solo_se_exige_con_acciones_continuas():
    cfg = {"bc": {"capas_ocultas": [8], "activacion": "tanh"}}
    assert construir_politica_bc(cfg, OBSERVACION, DISCRETO).log_std is None
    with pytest.raises(ValueError, match="bc.log_std_inicial"):
        construir_politica_bc(cfg, OBSERVACION, CONTINUO)


def test_los_valores_no_validos_de_la_config_llegan_como_valueerror():
    with pytest.raises(ValueError, match="capas_ocultas"):
        construir_politica_bc(_cfg(capas_ocultas=[100, 0]), OBSERVACION, DISCRETO)
    with pytest.raises(ValueError, match="log_std_inicial"):
        construir_politica_bc(_cfg(log_std_inicial=None), OBSERVACION, CONTINUO)


# --- DatasetTransiciones ----------------------------------------------------------

@pytest.mark.parametrize("inicio,dtype", [
    (inicio, dtype) for inicio in (0, -1, 3) for dtype in (np.int64, np.int32, np.uint8)
    if not (inicio < 0 and dtype is np.uint8)  # uint8 no representa acciones negativas
])
def test_dataset_discreto_resta_el_start_y_da_int64(inicio, dtype):
    acciones = (np.array([0, 1, 2, 1]) + inicio).astype(dtype)
    dataset = DatasetTransiciones(_lote(np.ones((4, 4), np.float64), acciones),
                                  spaces.Discrete(3, start=inicio))
    assert dataset.acciones.dtype == torch.int64
    assert dataset.acciones.tolist() == [0, 1, 2, 1]
    assert dataset.estados.dtype == torch.float32 and dataset.estados.shape == (4, 4)


def test_dataset_continuo_da_float32_con_los_mismos_valores():
    acciones = np.array([[0.5, -1.0], [0.25, 0.0]], dtype=np.float64)
    dataset = DatasetTransiciones(_lote(np.zeros((2, 4)), acciones), CONTINUO)
    assert dataset.acciones.dtype == torch.float32
    assert torch.equal(dataset.acciones, torch.tensor(acciones, dtype=torch.float32))


def test_dataset_devuelve_pares_y_conserva_ids():
    estados = np.arange(12, dtype=np.float32).reshape(3, 4)
    dataset = DatasetTransiciones(_lote(estados, [1, 0, 1], ids=[7, 7, 9]), DISCRETO)
    assert len(dataset) == 3
    estado, accion = dataset[2]
    assert torch.equal(estado, torch.tensor(estados[2])) and accion.item() == 1
    assert dataset.ids_episodio.tolist() == [7, 7, 9]


def test_dataset_copia_los_arrays_del_lote():
    estados = np.zeros((2, 4), dtype=np.float32)
    acciones = np.zeros((2, 2), dtype=np.float32)
    dataset = DatasetTransiciones(_lote(estados, acciones), CONTINUO)
    estados[:] = 5
    acciones[:] = 5
    assert dataset.estados.sum() == 0 and dataset.acciones.sum() == 0


@pytest.mark.parametrize("estados,acciones,espacio,mensaje", [
    (np.zeros((0, 4)), np.zeros(0, np.int64), DISCRETO, "N > 0"),
    (np.zeros(4), np.zeros(1, np.int64), DISCRETO, "N > 0"),
    (np.zeros((2, 4), dtype=bool), np.zeros(2, np.int64), DISCRETO, "números"),
    (np.full((2, 4), np.nan), np.zeros(2, np.int64), DISCRETO, "finitos"),
    (np.zeros((2, 4)), np.zeros(3, np.int64), DISCRETO, "enteros"),
    (np.zeros((2, 4)), np.zeros(2, np.float32), DISCRETO, "enteros"),
    (np.zeros((2, 4)), np.zeros((2, 1), np.int64), DISCRETO, "enteros"),
    (np.zeros((2, 4)), np.array([0, 2]), DISCRETO, "fuera"),
    (np.zeros((2, 4)), np.array([-1, 0]), DISCRETO, "fuera"),
    (np.zeros((2, 4)), np.zeros(2, np.float32), CONTINUO, "reales"),
    (np.zeros((2, 4)), np.zeros((2, 3), np.float32), CONTINUO, "reales"),
    (np.zeros((2, 4)), np.zeros((2, 2), np.int64), CONTINUO, "reales"),
    (np.zeros((2, 4)), np.array([[0.0, np.inf], [0.0, 0.0]]), CONTINUO, "finitas"),
    (np.zeros((2, 4)), np.zeros(2, np.int64), spaces.MultiDiscrete([2]), "no compatible"),
    (np.zeros((2, 4)), np.zeros((2, 2, 1), np.float32),
     spaces.Box(-1, 1, (2, 1), np.float32), "no compatible"),
])
def test_dataset_rechaza_lotes_no_validos(estados, acciones, espacio, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        DatasetTransiciones(_lote(estados, acciones), espacio)


def test_dataset_rechaza_lo_que_no_es_un_lote_o_ids_incoherentes():
    with pytest.raises(ValueError, match="LoteTransiciones"):
        DatasetTransiciones((np.zeros((2, 4)), np.zeros(2)), DISCRETO)
    with pytest.raises(ValueError, match="ids_episodio"):
        DatasetTransiciones(_lote(np.zeros((2, 4)), [0, 1], ids=[0]), DISCRETO)


# --- construir_loaders_bc ---------------------------------------------------------

@TIPOS
@pytest.mark.parametrize("cantidad", [1, 4, 7, 10])
def test_loaders_separan_las_partidas_del_subconjunto_y_de_validacion(cantidad, continua):
    trayectorias = _quince(continua)
    splits = _splits(trayectorias)
    train, val = construir_loaders_bc(trayectorias, splits, cantidad,
                                      CONTINUO if continua else DISCRETO, CFG, crear_generador(1))
    ids_train = set(train.dataset.ids_episodio.tolist())
    ids_val = set(val.dataset.ids_episodio.tolist())
    assert ids_train == set(splits.subconjuntos[cantidad])
    assert ids_val == set(splits.validacion)
    assert not ids_train & ids_val
    # Con frecuencia 1 entran todos los pasos de cada partida.
    por_id = {t.id_episodio: t for t in trayectorias}
    assert len(train.dataset) == sum(por_id[i].numero_pasos for i in splits.subconjuntos[cantidad])
    assert len(val.dataset) == sum(por_id[i].numero_pasos for i in splits.validacion)


@pytest.mark.parametrize("frecuencia", [1, 2, 3, 20])
def test_loaders_sortean_validacion_y_luego_train_con_el_mismo_generador(frecuencia):
    trayectorias = _quince()
    splits = _splits(trayectorias)
    train, val = construir_loaders_bc(trayectorias, splits, 4, DISCRETO,
                                      _cfg(frecuencia_pares=frecuencia), crear_generador(3))
    por_id = {t.id_episodio: t for t in trayectorias}
    generador = crear_generador(3)
    esperado_val = submuestrear_pares([por_id[i] for i in splits.validacion],
                                      frecuencia, generador)
    esperado_train = submuestrear_pares([por_id[i] for i in splits.subconjuntos[4]],
                                        frecuencia, generador)
    for loader, esperado in ((train, esperado_train), (val, esperado_val)):
        assert np.array_equal(loader.dataset.estados.numpy(), esperado.estados)
        assert np.array_equal(loader.dataset.acciones.numpy(), esperado.acciones)
        assert np.array_equal(loader.dataset.ids_episodio, esperado.ids_episodio)


def _pares_por_cantidad(frecuencia: int, semilla: int):
    """Estados de train y validación para cada cantidad, con la misma semilla.

    En las partidas inventadas el estado del paso t vale 100 * id + t, así que
    comparar estados equivale a comparar qué paso de qué partida se tomó.
    """
    trayectorias = _quince()
    splits = _splits(trayectorias)
    pares = {}
    for cantidad in (1, 4, 7, 10):
        train, val = construir_loaders_bc(trayectorias, splits, cantidad, DISCRETO,
                                          _cfg(frecuencia_pares=frecuencia),
                                          crear_generador(semilla))
        pares[cantidad] = (train.dataset.estados, val.dataset.estados)
    return pares


@pytest.mark.parametrize("frecuencia", [2, 3, 20])
@pytest.mark.parametrize("semilla", [0, 1, 7])
def test_validacion_tiene_los_mismos_pares_para_cualquier_cantidad(frecuencia, semilla):
    """Regresión: al sortear train primero, validación cambiaba con la cantidad."""
    pares = _pares_por_cantidad(frecuencia, semilla)
    _, val_1 = pares[1]
    for cantidad in (4, 7, 10):
        assert torch.equal(pares[cantidad][1], val_1)


@pytest.mark.parametrize("frecuencia", [2, 3, 20])
@pytest.mark.parametrize("semilla", [0, 1, 7])
def test_los_pares_de_train_estan_anidados_entre_cantidades(frecuencia, semilla):
    """Cada subconjunto aporta los mismos pares que el anterior y añade otros."""
    pares = _pares_por_cantidad(frecuencia, semilla)
    for menor, mayor in ((1, 4), (4, 7), (7, 10)):
        train_menor, train_mayor = pares[menor][0], pares[mayor][0]
        assert len(train_menor) < len(train_mayor)
        assert torch.equal(train_mayor[:len(train_menor)], train_menor)


def test_loaders_usan_batch_size_y_solo_barajan_train():
    trayectorias = _quince()
    train, val = construir_loaders_bc(trayectorias, _splits(trayectorias), 10, DISCRETO,
                                      _cfg(batch_size=5), crear_generador(0))
    assert train.batch_size == val.batch_size == 5
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)


def test_loaders_traducen_el_start_de_las_acciones_discretas():
    trayectorias = [_partida(i, inicio_accion=-1) for i in range(15)]
    train, val = construir_loaders_bc(trayectorias, _splits(trayectorias), 10,
                                      spaces.Discrete(2, start=-1), CFG, crear_generador(0))
    assert set(train.dataset.acciones.tolist()) == set(val.dataset.acciones.tolist()) == {0, 1}


@pytest.mark.parametrize("cantidad", [2, 0, 11, 4.0, True, "4", None])
def test_loaders_rechazan_cantidades_sin_subconjunto(cantidad):
    trayectorias = _quince()
    with pytest.raises(ValueError, match="subconjunto"):
        construir_loaders_bc(trayectorias, _splits(trayectorias), cantidad, DISCRETO, CFG,
                             crear_generador(0))


def test_loaders_rechazan_splits_de_otras_partidas():
    trayectorias = _quince()
    splits = _splits(trayectorias)
    with pytest.raises(ValueError, match="faltan"):
        construir_loaders_bc(trayectorias[:-1], splits, 4, DISCRETO, CFG, crear_generador(0))
    with pytest.raises(ValueError, match="Trayectoria"):
        construir_loaders_bc("partidas", splits, 4, DISCRETO, CFG, crear_generador(0))


@pytest.mark.parametrize("cfg", [
    {}, None, {"bc": {"batch_size": 8}}, {"bc": {"frecuencia_pares": 1}},
])
def test_loaders_rechazan_configuracion_incompleta(cfg):
    trayectorias = _quince()
    with pytest.raises(ValueError, match="bc.frecuencia_pares"):
        construir_loaders_bc(trayectorias, _splits(trayectorias), 4, DISCRETO, cfg,
                             crear_generador(0))


@pytest.mark.parametrize("cambio,mensaje", [
    ({"batch_size": 0}, "batch_size"), ({"batch_size": True}, "batch_size"),
    ({"batch_size": 8.0}, "batch_size"), ({"frecuencia_pares": 0}, "frecuencia"),
])
def test_loaders_rechazan_valores_no_validos(cambio, mensaje):
    trayectorias = _quince()
    with pytest.raises(ValueError, match=mensaje):
        construir_loaders_bc(trayectorias, _splits(trayectorias), 4, DISCRETO, _cfg(**cambio),
                             crear_generador(0))


def test_loaders_rechazan_acciones_que_no_encajan_con_el_espacio():
    trayectorias = _quince(continua=True)
    with pytest.raises(ValueError, match="Discrete"):
        construir_loaders_bc(trayectorias, _splits(trayectorias), 4, DISCRETO, CFG,
                             crear_generador(0))


# --- evaluar_bc ------------------------------------------------------------------

@TIPOS
def test_evaluar_bc_promedia_por_par_con_batches_desiguales(continua):
    estados, acciones = _datos_aprendibles(continua, n=23)
    espacio = CONTINUO if continua else DISCRETO
    politica = _politica(continua)
    por_batches = evaluar_bc(politica, _loader(estados, acciones, espacio, batch_size=5))
    de_golpe = evaluar_bc(politica, _loader(estados, acciones, espacio, batch_size=23))
    assert por_batches.keys() == de_golpe.keys() == (
        {"perdida", "error_cuadratico"} if continua else {"perdida", "acierto"})
    for clave in por_batches:
        assert por_batches[clave] == pytest.approx(de_golpe[clave], rel=1e-5)


def test_evaluar_bc_calcula_acierto_y_error_cuadratico_exactos():
    discreta = _politica(False, capas=())
    with torch.no_grad():
        discreta.red[0].weight.copy_(torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]]))
        discreta.red[0].bias.zero_()
    estados = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [2, 0, 0, 0], [-2, 0, 0, 0]], np.float32)
    # La acción más probable es 0 si x0 > 0; acierta en 3 de 4 pares.
    assert evaluar_bc(discreta, _loader(estados, [0, 1, 0, 0], DISCRETO))["acierto"] == 0.75

    continua = _politica(True, capas=())
    with torch.no_grad():
        continua.red[0].weight.zero_()
        continua.red[0].bias.copy_(torch.tensor([1.0, -1.0]))
    acciones = np.array([[1, -1], [3, -1], [1, 1], [1, -1]], np.float32)
    # Errores al cuadrado por par, sumados por componente: 0, 4, 4, 0.
    metricas = evaluar_bc(continua, _loader(estados, acciones, CONTINUO))
    assert metricas["error_cuadratico"] == pytest.approx(2.0)


@TIPOS
def test_evaluar_bc_pone_modo_evaluacion_y_no_calcula_gradientes(continua):
    politica = _politica(continua).train()
    estados, acciones = _datos_aprendibles(continua, n=8)
    evaluar_bc(politica, _loader(estados, acciones, CONTINUO if continua else DISCRETO))
    assert not politica.training
    assert all(p.grad is None for p in politica.parameters())


def test_evaluar_bc_rechaza_un_loader_vacio():
    vacio = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64)))
    with pytest.raises(ValueError, match="no tiene pares"):
        evaluar_bc(_politica(False), vacio)


# --- entrenar_bc -------------------------------------------------------------------

@TIPOS
def test_entrenar_bc_aprende_una_relacion_sencilla(continua):
    fijar_seed(0)
    espacio = CONTINUO if continua else DISCRETO
    train = _loader(*_datos_aprendibles(continua, semilla=0), espacio, barajar=True)
    val = _loader(*_datos_aprendibles(continua, n=128, semilla=1), espacio)
    politica = _politica(continua, capas=(32,))
    antes = evaluar_bc(politica, val)
    resultado = entrenar_bc(politica, train, val, _cfg(epocas_maximas=60, paciencia=60))
    despues = evaluar_bc(politica, val)
    assert despues["perdida"] < antes["perdida"]
    if continua:
        assert despues["error_cuadratico"] < 0.1 * antes["error_cuadratico"]
    else:
        assert despues["acierto"] > 0.9
    assert resultado["val_perdida"] == pytest.approx(despues["perdida"], rel=1e-6)


@pytest.mark.parametrize("paciencia", [1, 2, 5])
def test_entrenar_bc_para_tras_paciencia_epocas_sin_mejora(paciencia):
    """Con learning rate 0 la pérdida de validación no cambia: la mejor época es la 1."""
    estados, acciones = _datos_aprendibles(False, n=16)
    loader = _loader(estados, acciones, DISCRETO)
    resultado = entrenar_bc(_politica(False), loader, loader,
                            _cfg(learning_rate=0, paciencia=paciencia, epocas_maximas=50))
    assert resultado["mejor_epoca"] == 1
    assert resultado["epocas"] == 1 + paciencia


@pytest.mark.parametrize("epocas_maximas", [1, 3])
def test_entrenar_bc_no_supera_las_epocas_maximas(epocas_maximas):
    estados, acciones = _datos_aprendibles(False, n=16)
    loader = _loader(estados, acciones, DISCRETO)
    resultado = entrenar_bc(_politica(False), loader, loader,
                            _cfg(epocas_maximas=epocas_maximas, paciencia=100))
    assert resultado["epocas"] == epocas_maximas


@TIPOS
def test_entrenar_bc_restaura_los_pesos_de_la_mejor_epoca(continua):
    """Validación con las acciones contrarias a train: empeora al aprender train."""
    fijar_seed(0)
    espacio = CONTINUO if continua else DISCRETO
    estados, acciones = _datos_aprendibles(continua, n=64)
    contrarias = -acciones if continua else 1 - acciones
    run = RunFalso()
    politica = _politica(continua, capas=(32,))
    resultado = entrenar_bc(politica, _loader(estados, acciones, espacio, barajar=True),
                            _loader(estados, contrarias, espacio),
                            _cfg(epocas_maximas=30, paciencia=3), run)
    clave = _criterio(continua)
    valores = run.valores(f"bc/val/{clave}")
    assert len(valores) == resultado["epocas"]
    assert resultado["mejor_epoca"] == 1 + int(np.argmin(valores))
    assert resultado[f"val_{clave}"] == pytest.approx(min(valores))
    assert valores[-1] > min(valores)
    final = evaluar_bc(politica, _loader(estados, contrarias, espacio))
    assert final[clave] == pytest.approx(min(valores), rel=1e-6)


@TIPOS
def test_entrenar_bc_es_reproducible_con_la_misma_semilla(continua):
    espacio = CONTINUO if continua else DISCRETO

    def entrenar_una_vez():
        fijar_seed(7)
        politica = _politica(continua)
        train = _loader(*_datos_aprendibles(continua, n=64), espacio, batch_size=8, barajar=True)
        val = _loader(*_datos_aprendibles(continua, n=32, semilla=1), espacio)
        return politica, entrenar_bc(politica, train, val, _cfg(epocas_maximas=5, paciencia=5))

    a, resultado_a = entrenar_una_vez()
    b, resultado_b = entrenar_una_vez()
    assert resultado_a == resultado_b
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


@TIPOS
def test_entrenar_bc_registra_train_y_validacion_con_pasos_crecientes(continua):
    espacio = CONTINUO if continua else DISCRETO
    estados, acciones = _datos_aprendibles(continua, n=20)
    run = RunFalso()
    entrenar_bc(_politica(continua), _loader(estados, acciones, espacio, batch_size=8),
                _loader(estados, acciones, espacio), _cfg(epocas_maximas=2, paciencia=5), run)
    pasos_train = [paso for m, paso in run.registros if "bc/train/perdida" in m]
    assert pasos_train == [1, 2, 3, 4, 5, 6]
    clave = "bc/val/error_cuadratico" if continua else "bc/val/acierto"
    assert [paso for m, paso in run.registros if clave in m] == [3, 6]


@TIPOS
def test_entrenar_bc_elige_por_perdida_en_discretas_y_por_error_cuadratico_en_continuas(
        continua, monkeypatch):
    """Las dos métricas discrepan a propósito: la pérdida es mínima en la época 2 y
    el error cuadrático en la 4. Las discretas no tienen error cuadrático."""
    perdidas = iter([3.0, 1.0, 2.0, 2.5, 2.6, 2.7])
    errores = iter([4.0, 3.0, 2.0, 1.0, 1.5, 1.6])

    def evaluar_falso(politica, loader):
        segunda = ("error_cuadratico", next(errores)) if continua else ("acierto", 0.5)
        return {"perdida": next(perdidas), segunda[0]: segunda[1]}

    monkeypatch.setattr("src.training.loop.evaluar_bc", evaluar_falso)
    estados, acciones = _datos_aprendibles(continua, n=8)
    loader = _loader(estados, acciones, CONTINUO if continua else DISCRETO)
    resultado = entrenar_bc(_politica(continua), loader, loader,
                            _cfg(epocas_maximas=6, paciencia=2))
    assert resultado["mejor_epoca"] == (4 if continua else 2)
    assert resultado["epocas"] == (6 if continua else 4)


def test_entrenar_bc_normaliza_solo_con_los_estados_de_train():
    """Validación con otra escala: si se usara, la media y la escala serían otras."""
    estados_train, acciones_train = _datos_aprendibles(False, n=40, semilla=0)
    estados_train = estados_train * 3 + 10
    estados_val, acciones_val = _datos_aprendibles(False, n=40, semilla=1)
    politica = _politica(False)
    entrenar_bc(politica, _loader(estados_train, acciones_train, DISCRETO, batch_size=7,
                                  barajar=True),
                _loader(estados_val * 100 - 50, acciones_val, DISCRETO),
                _cfg(epocas_maximas=1))
    datos = estados_train.astype(np.float64)
    assert np.allclose(politica.media.numpy(), datos.mean(axis=0), rtol=1e-6)
    assert np.allclose(politica.escala.numpy(), datos.std(axis=0), rtol=1e-6)


def test_entrenar_bc_sin_normalizar_deja_la_normalizacion_como_estaba():
    estados, acciones = _datos_aprendibles(False, n=16)
    loader = _loader(estados * 5 + 3, acciones, DISCRETO)
    politica = _politica(False)
    entrenar_bc(politica, loader, loader,
                _cfg(normalizar_observaciones=False, epsilon_normalizacion=None, epocas_maximas=1))
    assert torch.equal(politica.media, torch.zeros(4)) and torch.equal(politica.escala, torch.ones(4))


@TIPOS
def test_con_normalizacion_el_entrenamiento_no_depende_de_la_escala_de_los_estados(continua):
    """Cambiar unidades y origen de cada componente da el mismo entrenamiento."""
    espacio = CONTINUO if continua else DISCRETO
    estados, acciones = _datos_aprendibles(continua, n=64)
    estados_val, acciones_val = _datos_aprendibles(continua, n=32, semilla=1)
    escala = np.array([1000.0, 0.001, 1.0, 50.0], dtype=np.float32)
    origen = np.array([-300.0, 0.5, 0.0, 20.0], dtype=np.float32)

    def entrenar(transformar):
        fijar_seed(3)
        politica = _politica(continua)
        resultado = entrenar_bc(
            politica, _loader(transformar(estados), acciones, espacio, batch_size=16, barajar=True),
            _loader(transformar(estados_val), acciones_val, espacio),
            _cfg(epocas_maximas=5, paciencia=5))
        return politica, resultado

    original, resultado_original = entrenar(lambda x: x)
    cambiada, resultado_cambiado = entrenar(lambda x: x * escala + origen)
    for clave, valor in resultado_original.items():
        assert resultado_cambiado[clave] == pytest.approx(valor, rel=1e-3, abs=1e-5)
    for po, pc in zip(original.parameters(), cambiada.parameters()):
        assert torch.allclose(po, pc, atol=1e-3)


def test_entrenar_bc_rechaza_un_loader_de_train_vacio():
    vacio = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64)))
    estados, acciones = _datos_aprendibles(False, n=8)
    val = _loader(estados, acciones, DISCRETO)
    for normalizar in (True, False):
        with pytest.raises(ValueError, match="train no tiene pares"):
            entrenar_bc(_politica(False), vacio, val, _cfg(normalizar_observaciones=normalizar))


def test_entrenar_bc_falla_si_la_perdida_no_es_finita():
    """Acciones finitas pero enormes: su cuadrado desborda float32 y la pérdida es inf."""
    enormes = DataLoader(TensorDataset(torch.zeros(4, 4), torch.full((4, 2), 1e30)))
    with pytest.raises(FloatingPointError, match="época 1"):
        entrenar_bc(_politica(True), enormes, enormes, CFG)


@pytest.mark.parametrize("normalizar,mensaje", [
    (True, "deben ser finitos"),  # lo detecta ajustar_normalizacion
    (False, "invalid values"),    # lo detecta torch.distributions
])
def test_entrenar_bc_rechaza_estados_nan(normalizar, mensaje):
    nan = DataLoader(TensorDataset(torch.full((4, 4), math.nan), torch.zeros(4, dtype=torch.int64)))
    with pytest.raises(ValueError, match=mensaje):
        entrenar_bc(_politica(False), nan, nan, _cfg(normalizar_observaciones=normalizar))


@pytest.mark.parametrize("cambio,mensaje", [
    ({"epocas_maximas": 0}, "epocas_maximas"), ({"epocas_maximas": 2.0}, "epocas_maximas"),
    ({"epocas_maximas": True}, "epocas_maximas"), ({"paciencia": 0}, "paciencia"),
    ({"paciencia": None}, "paciencia"), ({"optimizador": "lbfgs"}, "optimizador"),
    ({"optimizador": None}, "optimizador"), ({"optimizador": ["adam"]}, "optimizador"),
    ({"learning_rate": -0.1}, "learning_rate"), ({"learning_rate": math.nan}, "learning_rate"),
    ({"learning_rate": math.inf}, "learning_rate"), ({"learning_rate": "0.1"}, "learning_rate"),
    ({"learning_rate": True}, "learning_rate"),
    ({"normalizar_observaciones": None}, "normalizar_observaciones"),
    ({"normalizar_observaciones": 1}, "normalizar_observaciones"),
    ({"normalizar_observaciones": "true"}, "normalizar_observaciones"),
    ({"epsilon_normalizacion": 0}, "epsilon"), ({"epsilon_normalizacion": None}, "epsilon"),
])
def test_entrenar_bc_rechaza_configuraciones_no_validas(cambio, mensaje):
    estados, acciones = _datos_aprendibles(False, n=8)
    loader = _loader(estados, acciones, DISCRETO)
    with pytest.raises(ValueError, match=mensaje):
        entrenar_bc(_politica(False), loader, loader, _cfg(**cambio))


@pytest.mark.parametrize("cfg", [{}, None, {"bc": None}])
def test_entrenar_bc_rechaza_configuraciones_sin_seccion_bc(cfg):
    estados, acciones = _datos_aprendibles(False, n=8)
    loader = _loader(estados, acciones, DISCRETO)
    with pytest.raises(ValueError, match="bc"):
        entrenar_bc(_politica(False), loader, loader, cfg)


# --- entrenar_bc_subconjunto ------------------------------------------------------

def _espacio(continua: bool):
    return CONTINUO if continua else DISCRETO


@TIPOS
def test_entrenar_bc_subconjunto_equivale_a_componer_las_piezas(continua):
    """Partidas -> loaders -> política -> entrenamiento, con las mismas semillas."""
    trayectorias = _quince(continua)
    splits = _splits(trayectorias)
    cfg = _cfg(frecuencia_pares=2, epocas_maximas=4, paciencia=2)

    fijar_seed(0)
    politica, resumen = entrenar_bc_subconjunto(trayectorias, splits, 4, OBSERVACION,
                                                _espacio(continua), cfg, crear_generador(5))
    fijar_seed(0)
    train, val = construir_loaders_bc(trayectorias, splits, 4, _espacio(continua), cfg,
                                      crear_generador(5))
    esperada = construir_politica_bc(cfg, OBSERVACION, _espacio(continua))
    resultado = entrenar_bc(esperada, train, val, cfg)

    assert all(torch.equal(a, b) for a, b in zip(politica.state_dict().values(),
                                                 esperada.state_dict().values()))
    assert {k: resumen[k] for k in resultado} == resultado


@TIPOS
@pytest.mark.parametrize("cantidad", [1, 4, 7, 10])
def test_entrenar_bc_subconjunto_resume_cantidad_y_pares(cantidad, continua):
    trayectorias = _quince(continua)
    splits = _splits(trayectorias)
    fijar_seed(0)
    _, resumen = entrenar_bc_subconjunto(trayectorias, splits, cantidad, OBSERVACION,
                                         _espacio(continua), _cfg(epocas_maximas=1),
                                         crear_generador(0))
    por_id = {t.id_episodio: t for t in trayectorias}
    assert resumen["cantidad"] == cantidad
    # Con frecuencia 1 entran todos los pasos de cada partida.
    assert resumen["pares_train"] == sum(por_id[i].numero_pasos
                                         for i in splits.subconjuntos[cantidad])
    assert resumen["pares_validacion"] == sum(por_id[i].numero_pasos for i in splits.validacion)
    assert {"mejor_epoca", "epocas", "val_perdida"} <= resumen.keys()


def test_entrenar_bc_subconjunto_normaliza_con_las_partidas_del_subconjunto():
    trayectorias = _quince()
    splits = _splits(trayectorias)
    fijar_seed(0)
    politica, _ = entrenar_bc_subconjunto(trayectorias, splits, 4, OBSERVACION, DISCRETO,
                                          _cfg(epocas_maximas=1), crear_generador(0))
    por_id = {t.id_episodio: t for t in trayectorias}
    estados = np.concatenate([por_id[i].estados for i in splits.subconjuntos[4]]).astype(np.float64)
    assert np.allclose(politica.media.numpy(), estados.mean(axis=0), rtol=1e-6)
    assert np.allclose(politica.escala.numpy(), estados.std(axis=0), rtol=1e-6)


@pytest.mark.parametrize("ruta", ["configs/default.yaml", "configs/smoke.yaml"])
@TIPOS
def test_entrenar_bc_subconjunto_funciona_con_las_configs_del_proyecto(ruta, continua):
    cfg = yaml.safe_load((RAIZ / ruta).read_text(encoding="utf-8"))
    cfg["bc"]["epocas_maximas"] = min(cfg["bc"]["epocas_maximas"], 2)
    trayectorias = _quince(continua)
    fijar_seed(0)
    politica, resumen = entrenar_bc_subconjunto(trayectorias, _splits(trayectorias), 10,
                                                OBSERVACION, _espacio(continua), cfg,
                                                crear_generador(0))
    # frecuencia_pares es 10 y las partidas tienen 5 a 9 pasos: un par por partida.
    assert resumen["pares_train"] == 10 and resumen["pares_validacion"] == 5
    assert politica.continua == continua
    assert math.isfinite(resumen["val_perdida"])


@pytest.mark.parametrize("cantidad,espacio_observacion,espacio_accion,mensaje", [
    (3, OBSERVACION, DISCRETO, "subconjunto"),
    (4, spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32), DISCRETO, r"\[N, 5\]"),
    (4, OBSERVACION, CONTINUO, "Box"),
])
def test_entrenar_bc_subconjunto_propaga_los_errores(cantidad, espacio_observacion,
                                                     espacio_accion, mensaje):
    trayectorias = _quince()
    with pytest.raises(ValueError, match=mensaje):
        entrenar_bc_subconjunto(trayectorias, _splits(trayectorias), cantidad,
                                espacio_observacion, espacio_accion, CFG, crear_generador(0))
