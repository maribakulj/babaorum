import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from babaorum.reader.r6_arch import ARMS  # noqa: E402


def test_shapes_strides_and_padding_independence():
    torch.manual_seed(0)
    for name, cls in ARMS.items():
        m = cls(n_out=20).eval()
        W = 400
        x = torch.rand(2, 1, cls.h_in, W)
        x[1, :, :, 240:] = 1.0  # remplissage blanc, comme la mise en lot réelle
        widths = torch.tensor([W, 240])
        out = m(x, widths)
        lp, L = out[0], out[1]
        assert lp.shape[0] == 2 and lp.shape[2] == 20, name
        assert lp.shape[1] == W // cls.stride, (name, lp.shape)
        assert L.tolist() == [W // cls.stride, 240 // cls.stride], name
        assert cls.h_in // cls.stride >= 8, name  # cadence : ≥ 8 trames par hauteur de ligne
        # en lot, le remplissage blanc ne change la sortie qu'à la marge (BatchNorm :
        # la zone blanche n'est pas nulle) ; l'ÉVALUATION se fait ligne par ligne (règle R6)
        with torch.no_grad():
            alone = m(x[1:, :, :, :240], torch.tensor([240]))[0][0, : L[1]]
            assert (alone - lp[1, : L[1]]).abs().max() < 0.1, name


def test_b5_aux_and_training_step():
    m = ARMS["B5"](n_out=10)
    x = torch.rand(2, 1, 64, 200)
    lp, L, aux = m(x, torch.tensor([200, 160]), with_aux=True)
    tgt = torch.tensor([1, 2, 3, 4, 5])
    loss = torch.nn.functional.ctc_loss(lp.transpose(0, 1), tgt, L, torch.tensor([3, 2]))
    loss.backward()
    assert aux.shape == lp.shape
