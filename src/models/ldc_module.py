from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch import LongTensor, Tensor
from torchmetrics import MeanMetric, MinMetric


class LDCLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=0, reduction="none"
        )  # TODO: change to have sample weight, current is mean

        # metric objects for calculating and averaging accuracy across batches

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any, training: bool = False):
        loss_weights = None
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, date_tokens = batch
            x = (x, date_tokens)
        elif len(batch) == 4:
            x, y, date_tokens, loss_weights_or_date_text_tokens = batch

            if loss_weights_or_date_text_tokens.dtype == torch.long:
                x = (x, date_tokens, loss_weights_or_date_text_tokens)
            else:
                loss_weights = loss_weights_or_date_text_tokens
                x = (x, date_tokens)
        elif len(batch) == 5:
            x, y, year_tokens, month_tokens, day_tokens = batch
            x = (x, year_tokens, month_tokens, day_tokens)
        elif len(batch) == 6:
            x, y, date_tokens, year_tokens, month_tokens, day_tokens = batch
            x = (x, date_tokens, year_tokens, month_tokens, day_tokens)

        result = self.forward(x)
        if isinstance(result, tuple):
            demb_mask = None
            if len(result) == 4:
                logits, ref_demb, pred_demb, demb_mask = result
            elif len(result) == 3:
                logits, ref_demb, pred_demb = result
        else:
            logits = result

        loss: Tensor = self.criterion(logits.permute(0, 2, 1), y)
        if isinstance(result, tuple) and training:
            # for mse
            # demb_loss = torch.nn.functional.mse_loss(pred_demb, ref_demb, reduction="none") * 0.1

            # for cosine similarity
            demb_loss = (
                1 - torch.nn.functional.cosine_similarity(pred_demb, ref_demb, dim=1).unsqueeze(1)
            ) * 0.1

            if demb_mask is not None:
                demb_loss = demb_loss * demb_mask

            loss = torch.cat([demb_loss, loss], dim=1)
        if loss_weights is not None:
            loss = torch.einsum("bt,b -> bt", loss, loss_weights)
        loss = loss.sum() / (y != 0).sum()

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch, training=True)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_loss.compute()  # get current val acc
        self.val_loss_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def on_before_optimizer_step(self, optimizer):
        # inspect (unscaled) gradients here
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_warmup_steps=int(stepping_batches * 0.15),
                num_training_steps=stepping_batches,
            )
            interval = "epoch"
            if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                interval = "step"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LDCLitModule(None, None, None)
