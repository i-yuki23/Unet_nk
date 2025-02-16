import torch
from torch.utils.data import Dataset
import scipy.io
import os
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np


class MatDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform):
        """
        input_dir: 入力データのディレクトリ（例：'data/inputs'）
        target_dir: ターゲットデータのディレクトリ（例：'data/targets'）
        transform: オプションの前処理
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        input_files = sorted(os.listdir(self.input_dir))
        target_files = sorted(os.listdir(self.target_dir))

        self.input_images = []   # 個々の画像（Tensor）のリスト
        self.target_images = []  # 対応するターゲット画像のリスト

        for input_file, target_file in zip(input_files, target_files):
            input_path = os.path.join(self.input_dir, input_file)
            target_path = os.path.join(self.target_dir, target_file)
            
            input_data = scipy.io.loadmat(input_path)['v_dopdata']    # (256, 256, 1000)
            target_data = scipy.io.loadmat(target_path)['v_thetadata']  # (256, 256, 1000)
            
            input_data = input_data.transpose(2, 0, 1)    # (1000, 256, 256)
            target_data = target_data.transpose(2, 0, 1)  # (1000, 256, 256)
            
            num_images = input_data.shape[0]
            for i in range(num_images):

                img_np = input_data[i].astype(np.float32) 
                tgt_np = target_data[i].astype(np.float32)

                if self.transform is not None:
                    augmented = self.transform(image=img_np, mask=tgt_np)
                    img_np = augmented["image"]
                    tgt_np = augmented["mask"]

                img_np = np.expand_dims(img_np, axis=0)
                tgt_np = np.expand_dims(tgt_np, axis=0)
                    
                # NumPy配列 → PyTorchテンソルに変換
                img_tensor = torch.from_numpy(img_np)
                tgt_tensor = torch.from_numpy(tgt_np)
                    
                self.input_images.append(img_tensor)
                self.target_images.append(tgt_tensor)
                
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        return self.input_images[idx], self.target_images[idx]


class MatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_transform=None,
        val_test_transform=None,
        batch_size: int = 16,
        num_workers: int = 4,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.val_test_transform= val_test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        グループ単位（＝各ファイル単位）でデータ分割を行い、
        各split用に MatDatasetGroup を作成。
        """

        train_input_dir = os.path.join(self.train_dir, 'inputs')
        train_target_dir = os.path.join(self.train_dir, 'targets')
        val_input_dir = os.path.join(self.val_dir, 'inputs')
        val_target_dir = os.path.join(self.val_dir, 'targets')
        test_input_dir = os.path.join(self.test_dir, 'inputs')
        test_target_dir = os.path.join(self.test_dir, 'targets')

        self.train_dataset = MatDataset(
            input_dir=train_input_dir,
            target_dir=train_target_dir,
            transform=self.train_transform,
        )
        self.val_dataset = MatDataset(
            input_dir=val_input_dir,
            target_dir=val_target_dir,
            transform=self.val_test_transform
        )
        self.test_dataset = MatDataset(
            input_dir=test_input_dir,
            target_dir=test_target_dir,
            transform=self.val_test_transform,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
