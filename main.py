import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer
import torch.nn.functional as F
from datamodules.mat_datamodules import MatDataModule
from models.lit_unet import LitUNet
from torchvision.transforms import Compose
from loss_functions.WeightedMSELoss import WeightedMSELoss
import cv2
import albumentations as A


# class ZeroToMinus9999:
#     def __call__(self, tensor):
#         return torch.where(tensor == 0, torch.tensor(-9999.0, dtype=tensor.dtype), tensor)

# class MaxAbsScaler:
#     def __init__(self, max_abs_value):
#         self.max_abs_value = max_abs_value
#     def __call__(self, tensor):
#         return tensor / self.max_abs_value

# max_abs_value_input = 1.2040270566940308
# max_abs_value_tgt = 0.7881543636322021

# max_abs_scaler_input = MaxAbsScaler(max_abs_value=max_abs_value_input)
# max_abs_scaler_tgt = MaxAbsScaler(max_abs_value=max_abs_value_tgt)

train_transform = A.Compose([
    # A.HorizontalFlip(p=0.5),

    # A.Affine(
    # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    # border_mode=cv2.BORDER_CONSTANT,  # 定数補完モードに設定
    # value=0,                         # 補完値を0に設定
    # p=0.5
    # ),

    # A.Rotate(
    # limit=15,                        # 最大15度の回転
    # border_mode=cv2.BORDER_CONSTANT,  # 画像外は定数補完モード
    # value=0,                         # 補完値は0
    # p=0.5
    # ),
])

val_test_transform = A.Compose([
    
])

lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    dirpath='checkpoints/UnetPlusPlus/WeightedLoss/',
    filename='best-checkpoint-{epoch:02d}-{val_loss:.4f}',  
    save_top_k=1,  # ベストモデルのみ保存
    mode='min',  
    save_weights_only=True
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,  
    mode='min'  
)

trainer = Trainer(
    max_epochs=300,
    callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor] 
)


datamodule = MatDataModule(
        data_dir='data',
        train_transform=train_transform,
        val_test_transform=val_test_transform,
        batch_size=32,
        num_workers=4,
        seed=42
    )

model = LitUNet(learning_rate=1e-4, loss_fn=WeightedMSELoss())

trainer.fit(model, datamodule=datamodule)

trainer.test(model, datamodule=datamodule)

