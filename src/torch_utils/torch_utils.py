"""
This program is a copy and modification of the following program under Apache License Version 2.0.
- https://github.com/makaishi2/pythonlibs/blob/9f3d9314531d799803cc82565230822c794c69ee/torch_lib1/__init__.py

LICENSE:
- https://github.com/makaishi2/pythonlibs/blob/main/LICENSE
"""  # noqa: E501

from collections.abc import Callable, Iterable
from typing import Optional, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch_geometric.data
import torch_geometric.loader

from torch_utils.utils import check_tqdm

# for type hint
DataBatch = Union[torch_geometric.data.Batch, torch_geometric.data.Data]
DataLoader = Union[
    torch_geometric.loader.DataLoader,
    Iterable[DataBatch],
]
LossFn = Union[
    torch.nn.modules.loss._Loss,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    This class is a copy and modification of the following program under MIT License.

    Reference:
    - https://github.com/Bjarten/early-stopping-pytorch/blob/3a28f68adeb44eab97716694367cc8c19ab331fd/pytorchtools.py
    """  # noqa: E501

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: Optional[str] = None,
        trace_func: Callable[[str], None] = print,
    ) -> None:
        """
        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss improved.
            , by default 7
        verbose : bool, optional
            If True, prints a message for each validation loss improvement.
            , by default False
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an
            improvement., by default 0.0
        path : str, optional
            Path for the checkpoint to be saved to,
            like 'checkpoint.pt'. if None, not save.
            , by default None
        trace_func : Callable[[str], None], optional
            trace print function., by default print
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.val_loss_min = float("inf")

    def __call__(
        self,
        val_loss: float,
        model: torch.nn.Module,
    ) -> bool:
        """check early stopping

        Parameters
        ----------
        val_loss : float
            validation loss
        model : torch.nn.Module, optional
            PyTorch model to be saved., by default None

        Returns
        -------
        bool
            True if early stopping.
        """
        if val_loss < self.val_loss_min - self.delta:
            self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                "Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}).  "
                "Saving model ..."
            )
        if self.path is not None:
            torch.save(model.state_dict(), self.path)


# 損失関数値計算用
def eval_loss(
    loader: DataLoader,
    net: torch.nn.Module,
    criterion: LossFn,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    # DataLoaderから最初の1セットを取得する
    data = next(iter(loader))

    # デバイスの割り当て
    data = data.to(device)

    # 予測値の計算
    y_pred = net(data)

    #  損失値の計算
    loss = criterion(y_pred, data.y)

    return loss


# 学習用関数
def fit(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: LossFn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    num_epochs: int = 100,
    history: np.ndarray = np.empty((0, 3)),
    early_stopping: Optional[EarlyStopping] = None,
    silent: bool = False,
) -> np.ndarray:
    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_val = 0, 0

        # 訓練フェーズ
        net.train()

        # for inputs, labels in tqdm(train_loader):
        for data_train in check_tqdm(train_loader, silent=silent):
            # 1バッチあたりのデータ件数
            train_batch_size = len(data_train)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size

            # GPUヘ転送
            data_train = data_train.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            y_pred_train = net(data_train)

            # 損失計算
            loss = criterion(y_pred_train, data_train.y)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size

        # 予測フェーズ
        net.eval()

        for data_val in val_loader:
            # 1バッチあたりのデータ件数
            val_batch_size = len(data_val)
            # 1エポックあたりのデータ累積件数
            n_val += val_batch_size

            # GPUヘ転送
            data_val = data_val.to(device)

            # 予測計算
            y_pred_val = net(data_val)

            # 損失計算
            loss_val = criterion(y_pred_val, data_val.y)

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss += loss_val.item() * val_batch_size

        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_val
        # 結果表示
        print(
            f"Epoch [{(epoch+1)}/{num_epochs+base_epochs}], "
            f"loss: {avg_train_loss:.5f} "
            f"val_loss: {avg_val_loss:.5f}, "
        )
        # 記録
        item = np.array([epoch + 1, avg_train_loss, avg_val_loss])
        history = np.vstack((history, item))

        # early stopping
        if early_stopping is not None and early_stopping(
            val_loss=avg_val_loss, model=net
        ):
            break
    return history


# 学習ログ解析
def evaluate_history(history: np.ndarray) -> matplotlib.axes.Axes:
    # 損失と精度の確認
    print(f"初期状態: 損失: {history[0,2]:.5f}")
    print(f"最終状態: 損失: {history[-1,2]:.5f}")

    num_epochs = len(history)
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10

    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="w", figsize=(6.4, 4.8))
    ax: matplotlib.axes.Axes
    ax.plot(history[:, 0], history[:, 1], "b", label="train")
    ax.plot(history[:, 0], history[:, 2], "k", label="validation")
    ax.set_xticks(np.arange(0, num_epochs + 1, unit))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss history")
    ax.legend()
    fig.tight_layout()
    return ax


def torch_seed(seed: int = 123) -> None:
    """PyTorch乱数固定用

    Parameters
    ----------
    seed : int, optional
        , by default 123
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)  # for ARM
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
