"""La misma seed tiene que dar el mismo numero. Si este test falla, cualquier
comparacion entre experimentos del proyecto carece de valor."""

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
