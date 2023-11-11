from collections.abc import Sequence
from typing import SupportsIndex, Optional

import rdkit.Chem.rdchem
from joblib import Parallel, delayed
import torch.utils.data
import torch_geometric.data

from torch_utils.feat import mol2data


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mols: Sequence[rdkit.Chem.rdchem.Mol],
        y: Sequence[torch.Tensor],
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.mols = mols if hasattr(mols, "__len__") else tuple(mols)
        self.n_jobs = n_jobs
        self.y = (
            torch.tensor(y, dtype=torch.float32)
            if hasattr(y, "__len__")
            else tuple(y)
        )

        # check the length of mols and y
        assert len(self.mols) == len(self.y), (
            "The length of mols and y must be same."
            f"{len(self.mols)} != {len(self.y)}"
        )

        # convert rdkit.Chem.rdchem.Mol to torch_geometric.data.Data
        self._data: list[torch_geometric.data.Data] = Parallel(
            n_jobs=self.n_jobs
        )(
            delayed(mol2data)(
                mol=mol,
                y=_y.view(1, -1),
                use_chirality=False,
                use_partial_charge=False,
                use_edges=False,
            )
            for mol, _y in zip(self.mols, self.y)
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: SupportsIndex) -> torch_geometric.data.Data:
        return self._data[index]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)})"

    def __repr__(self) -> str:
        return str(self)
