import pytest
torch = pytest.importorskip("torch")
from agentic_rob.attacks import fgsm, pgd, bim, cw_l2, square_linf


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.f = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 3))
    def forward(self, x): return self.f(x)


def test_budgets_and_range():
    torch.manual_seed(0); m = Tiny().eval(); x = torch.rand(8, 3, 16, 16); y = torch.randint(0, 3, (8,)); eps = 8 / 255
    for xa in (fgsm(m, x, y, eps), pgd(m, x, y, eps, 2 / 255, 5), bim(m, x, y, eps, 2 / 255, 5), square_linf(m, x, y, eps, 20)):
        assert (xa - x).abs().max() <= eps + 1e-6 and xa.min() >= 0 and xa.max() <= 1
    xa = cw_l2(m, x, y, l2_eps=1.0, steps=5)
    assert (xa - x).flatten(1).norm(dim=1).max() <= 1.0 + 1e-4
