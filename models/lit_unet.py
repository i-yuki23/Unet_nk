import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import r2_score
from models.unetplusplus import UNetPlusPlus
from models.UnetWithAttention import UNetWithAttention

class LitUNet(pl.LightningModule):
    def __init__(self, loss_fn, learning_rate=1e-3):
        super(LitUNet, self).__init__()
        self.model = UNetWithAttention()
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.test_gt = []
        self.test_outputs = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)  # 渡された損失関数を使用
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.loss_fn(y_hat, y)

        y_true = y.detach().cpu().numpy().flatten()
        y_pred = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_true, y_pred)

        self.log('test_loss', mse_loss, prog_bar=True)
        self.log('test_r2', r2, prog_bar=True)

        self.test_gt.append(y)
        self.test_outputs.append(y_hat)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2),
            'monitor': 'val_loss',
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
