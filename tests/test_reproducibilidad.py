"""La misma seed tiene que dar el mismo numero. Si este test falla, cualquier
comparacion entre experimentos del proyecto carece de valor."""

import numpy as np
import pytest
import torch

from src.utils.seed import fijar_seed


def test_la_misma_seed_da_la_misma_politica_bc():
    from src.models.factory import PoliticaBC

    def crear(continua):
        return PoliticaBC(4, 2, continua, [100, 100], "tanh", 0.0 if continua else None)

    for continua in (False, True):
        fijar_seed(42)
        a = crear(continua)
        fijar_seed(42)
        b = crear(continua)
        fijar_seed(1234)
        c = crear(continua)
        assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
        assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


# --- estado de los generadores globales (checkpoints) -------------------------------------

def _tres_numeros() -> tuple:
    import random
    return random.random(), np.random.rand(), np.random.normal(), torch.rand(2).tolist()


def test_restaurar_el_estado_aleatorio_repite_los_mismos_numeros():
    import io

    from src.utils.seed import estado_aleatorio, fijar_seed, restaurar_estado_aleatorio
    fijar_seed(11)
    np.random.normal()  # deja un gaussiano guardado en NumPy
    estado = estado_aleatorio()
    esperados = _tres_numeros()
    fijar_seed(99)
    memoria = io.BytesIO()
    torch.save(estado, memoria)  # como en un checkpoint, con weights_only al leer
    memoria.seek(0)
    restaurar_estado_aleatorio(torch.load(memoria, weights_only=True))
    assert _tres_numeros() == esperados


def test_estado_aleatorio_no_gasta_numeros():
    from src.utils.seed import estado_aleatorio, fijar_seed
    fijar_seed(4)
    esperados = _tres_numeros()
    fijar_seed(4)
    estado_aleatorio()
    assert _tres_numeros() == esperados


@pytest.mark.parametrize("cambio", [
    lambda e: e.pop("torch"), lambda e: e.pop("python"), lambda e: e.pop("numpy"),
    lambda e: e.update(torch=[1, 2, 3]), lambda e: e.update(torch=torch.zeros(3)),
    lambda e: e["numpy"].update(estado=torch.full((624,), -1)),
    lambda e: e["numpy"].update(estado=torch.full((624,), 2**33)),
    lambda e: e["numpy"].update(clave="otro"), lambda e: e.update(python=(3,)),
    lambda e: e.update(numpy=None),
], ids=["sin_torch", "sin_python", "sin_numpy", "torch_lista", "torch_corto",
        "numpy_negativo", "numpy_grande", "numpy_clave", "python_corto", "numpy_none"])
def test_un_estado_aleatorio_que_no_vale(cambio):
    from src.utils.seed import estado_aleatorio, restaurar_estado_aleatorio
    estado = estado_aleatorio()
    cambio(estado)
    with pytest.raises(ValueError, match="Estado aleatorio no válido"):
        restaurar_estado_aleatorio(estado)
