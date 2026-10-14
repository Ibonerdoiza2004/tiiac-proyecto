"""Tests de forma: la primera linea de defensa del codigo de ML.

Un fallo de formas no da error: da un modelo que entrena sobre basura. Estos
tests son baratos y detectan la mayoria.
"""

import pytest
import torch

from src.models.factory import construir_modelo

CFG = {"modelo": {"nombre": "mi_modelo"}}


@pytest.mark.parametrize("nombre", ["bc", "gail", "", None])
def test_construir_modelo_rechaza_nombres_desconocidos(nombre):
    with pytest.raises(ValueError, match="modelo desconocido"):
        construir_modelo({"modelo": {"nombre": nombre}})


def test_forward_devuelve_la_forma_esperada():
    modelo = construir_modelo(CFG)
    x = torch.randn(4, 32)
    assert modelo(x).shape == (4, 10)


def test_el_batch_no_se_mezcla():
    """Si el modelo mezcla el eje de batch, dos entradas distintas dan la
    misma salida. Es el fallo silencioso mas caro que existe."""
    modelo = construir_modelo(CFG).eval()
    x = torch.randn(8, 32)
    with torch.no_grad():
        juntas = modelo(x)
        sueltas = torch.cat([modelo(x[i:i + 1]) for i in range(8)])
    assert torch.allclose(juntas, sueltas, atol=1e-5)


def test_los_gradientes_llegan_a_todos_los_parametros():
    """Un parametro con gradiente None esta desconectado del grafo: existe,
    ocupa memoria y no aprende."""
    modelo = construir_modelo(CFG)
    modelo(torch.randn(4, 32)).sum().backward()
    sin_gradiente = [n for n, p in modelo.named_parameters()
                     if p.requires_grad and p.grad is None]
    assert not sin_gradiente, f"parametros desconectados: {sin_gradiente}"



# --- PoliticaBC -------------------------------------------------------------
# Arquitecturas elegidas para cubrir: la del paper (2 x 100), una sola capa,
# capas de tamaños distintos, política lineal (sin capas) y tamaños mínimos
# (estado y acción de 1 componente), cada una con acciones discretas y continuas.
# CartPole es dim 4 y 2 acciones; Reacher, dim 11 (aquí 10) y 2 continuas.

import pytest  # noqa: E402

from src.models.factory import PoliticaBC  # noqa: E402

ARQUITECTURAS = [
    (4, 2, [100, 100]),
    (6, 3, [64]),
    (8, 5, [32, 16, 8]),
    (4, 2, []),
    (1, 1, [1]),
]
TIPOS = pytest.mark.parametrize("continua", [False, True], ids=["discreta", "continua"])


def _politica(dim: int, n: int, capas: list[int], continua: bool,
              activacion: str = "tanh") -> PoliticaBC:
    """Política con la capa de salida aleatoria.

    La inicialización original pone la salida a cero: todas las filas darían lo
    mismo y los tests de batch y gradientes no detectarían nada.
    """
    politica = PoliticaBC(dim, n, continua, capas, activacion, 0.0 if continua else None)
    with torch.no_grad():
        politica.red[-1].weight.normal_()
        politica.red[-1].bias.normal_()
    return politica


def _acciones(politica: PoliticaBC, batch: int) -> torch.Tensor:
    if politica.continua:
        return torch.randn(batch, politica.dim_accion)
    return torch.randint(0, politica.dim_accion, (batch,))


@TIPOS
@pytest.mark.parametrize("dim,n,capas", ARQUITECTURAS)
@pytest.mark.parametrize("batch", [0, 1, 7])
def test_politica_bc_devuelve_una_salida_por_componente_de_accion(dim, n, capas, continua, batch):
    politica = _politica(dim, n, capas, continua)
    estados = torch.randn(batch, dim)
    assert politica(estados).shape == (batch, n)
    assert politica(estados).dtype == torch.float32
    assert politica.log_verosimilitud(estados, _acciones(politica, batch)).shape == (batch,)


@TIPOS
@pytest.mark.parametrize("dim,n,capas", ARQUITECTURAS)
def test_politica_bc_la_distribucion_tiene_los_ejes_correctos(dim, n, capas, continua):
    distribucion = _politica(dim, n, capas, continua).distribucion(torch.randn(5, dim))
    assert distribucion.batch_shape == (5,)
    assert distribucion.event_shape == ((n,) if continua else ())
    assert distribucion.sample().shape == ((5, n) if continua else (5,))


@TIPOS
@pytest.mark.parametrize("dim,n,capas", ARQUITECTURAS)
def test_politica_bc_no_mezcla_el_batch(dim, n, capas, continua):
    politica = _politica(dim, n, capas, continua).eval()
    x = torch.randn(6, dim)
    a = _acciones(politica, 6)
    with torch.no_grad():
        juntas = politica.log_verosimilitud(x, a)
        sueltas = torch.cat([politica.log_verosimilitud(x[i:i + 1], a[i:i + 1])
                             for i in range(6)])
    assert torch.allclose(juntas, sueltas, atol=1e-6)


@TIPOS
@pytest.mark.parametrize("dim,n,capas", ARQUITECTURAS)
@pytest.mark.parametrize("activacion", ["tanh", "relu"])
def test_politica_bc_los_gradientes_llegan_a_todos_los_parametros(dim, n, capas, continua,
                                                                  activacion):
    """Incluye log_std: con acciones continuas solo recibe gradiente por la pérdida."""
    politica = _politica(dim, n, capas, continua, activacion)
    politica.log_verosimilitud(torch.randn(4, dim), _acciones(politica, 4)).sum().backward()
    sin_gradiente = [nombre for nombre, p in politica.named_parameters()
                     if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not sin_gradiente, f"parametros desconectados: {sin_gradiente}"
    assert ("log_std" in dict(politica.named_parameters())) == continua


@TIPOS
@pytest.mark.parametrize("dim,n,capas", ARQUITECTURAS)
def test_politica_bc_tiene_las_capas_declaradas(dim, n, capas, continua):
    lineales = [m for m in _politica(dim, n, capas, continua).modules()
                if isinstance(m, torch.nn.Linear)]
    tamanos = [dim, *capas, n]
    assert [(m.in_features, m.out_features) for m in lineales] == list(zip(tamanos, tamanos[1:]))


@pytest.mark.parametrize("activacion,clase", [("tanh", torch.nn.Tanh), ("relu", torch.nn.ReLU)])
def test_politica_bc_usa_la_activacion_pedida_entre_capas(activacion, clase):
    politica = _politica(4, 2, [10, 10], False, activacion)
    tipos = [type(m) for m in politica.red]
    assert tipos == [torch.nn.Linear, clase, torch.nn.Linear, clase, torch.nn.Linear]


@TIPOS
@pytest.mark.parametrize("forma", [(4,), (2, 3), (2, 5), (2, 4, 1), (0,)])
def test_politica_bc_rechaza_estados_con_forma_incorrecta(forma, continua):
    with pytest.raises(ValueError, match=r"\[B, 4\]"):
        _politica(4, 2, [8], continua)(torch.randn(*forma))


@TIPOS
@pytest.mark.parametrize("dim,n,capas", ARQUITECTURAS)
def test_politica_bc_se_inicializa_como_el_codigo_original(dim, n, capas, continua):
    """Glorot uniforme en las capas ocultas, salida y sesgos a cero (policyopt/nn.py)."""
    politica = PoliticaBC(dim, n, continua, capas, "tanh", 0.0 if continua else None)
    lineales = [m for m in politica.red if isinstance(m, torch.nn.Linear)]
    for capa in lineales[:-1]:
        cota = (6 / (capa.in_features + capa.out_features)) ** 0.5
        assert capa.weight.abs().max() <= cota
        assert capa.weight.abs().max() > 0
    assert all(torch.count_nonzero(capa.bias) == 0 for capa in lineales)
    assert torch.count_nonzero(lineales[-1].weight) == 0
    # Política inicial: uniforme sobre las acciones o media cero.
    assert torch.count_nonzero(politica(torch.randn(3, dim))) == 0
