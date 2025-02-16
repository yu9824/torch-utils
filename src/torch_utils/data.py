import sys

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence
import os
import pickle
from dataclasses import dataclass
from typing import SupportsIndex, Union

import torch.utils.data
import torch_geometric.data
from rdkit import Chem

from torch_utils.feat import mol2data
from torch_utils.utils import check_tqdm


@dataclass
class GraphDataset(torch.utils.data.Dataset):
    _smiles_set: Sequence[str]
    _y: Sequence[float]
    dirpath_cache: Union[os.PathLike, str] = "./_cache"
    use_chirality: bool = False
    use_partial_charge: bool = False
    use_edges: bool = False

    def __post_init__(
        self,
    ) -> None:
        _kwargs_mol2data = dict(
            use_chirality=self.use_chirality,
            use_partial_charge=self.use_partial_charge,
            use_edges=self.use_edges,
        )
        self._dataset: Sequence[torch_geometric.data.Data] = []

        for _idx_sample, _smiles in check_tqdm(
            enumerate(self._smiles_set), total=len(self._smiles_set)
        ):
            _filepath_cache = os.path.join(
                self.dirpath_cache,
                str(_kwargs_mol2data),
                f"{_idx_sample}.pkl",
            )
            if os.path.isfile(_filepath_cache):
                with open(_filepath_cache, mode="rb") as f:
                    _data: torch_geometric.data.Data = pickle.load(f)
            else:
                _data: torch_geometric.data.Data = mol2data(
                    Chem.MolFromSmiles(_smiles), **_kwargs_mol2data
                )
                os.makedirs(os.path.dirname(_filepath_cache), exist_ok=True)
                with open(_filepath_cache, mode="wb") as f:
                    pickle.dump(_data, f)

            _data.y = torch.tensor(self._y[_idx_sample]).reshape(-1, 1)

            self._dataset.append(_data)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> torch_geometric.data.Data:
        return self._dataset[index]

    @property
    def y(self) -> torch.Tensor:
        return torch.cat((_data.y for _data in self._dataset), dim=0)

    @y.setter
    def y(self, _value) -> None:
        raise AttributeError
