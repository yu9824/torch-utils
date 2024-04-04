from typing import Any

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from torch_utils.torch_utils import DataBatch, LossFn

# TODO: yやy_predをlogに保存
# TODO: train loss


class LightningGCN(lightning.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: LossFn = torch.nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, databatch: DataBatch) -> Any:
        return self.net(databatch)

    def training_step(
        self,
        databatch: DataBatch,
    ) -> STEP_OUTPUT:
        # 予測計算
        y_pred_train = self(databatch)

        # 損失計算
        loss = self.criterion(y_pred_train, databatch.y)

        # 'on_epoch' などのタイミングについては下記サイトを参考
        # https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        # こうすると自動でepochごとに平均を取ってくれる
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            batch_size=len(databatch),
        )

        return loss

    def validation_step(
        self,
        databatch: DataBatch,
    ):
        # 予測計算
        y_pred_val = self(databatch)

        # 損失計算
        loss = self.criterion(y_pred_val, databatch.y)
        self.log("val_loss", loss, batch_size=len(databatch))

        return loss

    def test_step(self, databatch: DataBatch) -> STEP_OUTPUT:
        # 予測計算
        y_pred_test = self(databatch)

        # 損失計算
        loss = self.criterion(y_pred_test, databatch.y)
        self.log("test_loss", loss, batch_size=len(databatch))

        return loss

    def predict_step(self, databatch: DataBatch) -> STEP_OUTPUT:
        return self(databatch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer
