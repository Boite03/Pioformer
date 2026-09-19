from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from pioformer.data.argoverse1 import Argoverse1Dataset


class Argoverse1DataModule(LightningDataModule):
    def __init__(self, root: str, batch_size: int = 32, num_workers: int = 8,
                 local_radius: float = 50.0) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.local_radius = local_radius

    def setup(self, stage=None) -> None:
        if stage in (None, "fit"):
            self.train_set = Argoverse1Dataset(self.root, "train", self.local_radius)
            self.val_set = Argoverse1Dataset(self.root, "val", self.local_radius)
        elif stage == "validate":
            self.val_set = Argoverse1Dataset(self.root, "val", self.local_radius)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)
