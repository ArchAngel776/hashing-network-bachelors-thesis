from torch import Tensor, no_grad, device as dev
from torch.utils.data import DataLoader
from app.modules.HSDH import HSDH


@no_grad()
def extract_hashes(model: HSDH, source: DataLoader[tuple[Tensor, Tensor]], device: dev) -> tuple[Tensor, Tensor]: ...
