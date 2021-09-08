import random
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torchio as tio
from torch.utils.data import DataLoader


def train_test_split(idx, train_ratio=.8, replace=False, k_sample=None):
    n_test = int(len(idx) * (1 - train_ratio))

    test_idx = random.sample(idx, n_test)
    train_idx = [i for i in idx if i not in test_idx]

    if replace:
        k_sample = k_sample if k_sample else len(train_idx)
        train_idx = random.choices(train_idx, k=k_sample)

    return train_idx, test_idx


class MRIDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: list,
                 datasets: list,
                 preprocessing_pipeline,
                 augmentation_pipeline,
                 postprocessing_pipeline,
                 specific_pipeline,
                 labels_names: list,
                 batch_size: int = 8,
                 train_val_test_idx: Optional[list] = None,
                 train_ratio: float = .8,
                 k_sample: Optional[int] = None,
                 replace: bool = False,
                 volume_pattern: list = ["**/tse*right*.nii.gz"],
                 label_pattern: list = ["seg*hippocampus_right*.nii.gz"],
                 ca_type: list = ["1/2/3"],
                 num_workers: int = 4,
                 pin_memory: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.datasets = datasets
        self.batch_size = batch_size
        self.train_val_test_idx = train_val_test_idx
        self.train_ratio = train_ratio
        self.k_sample = k_sample
        self.replace = replace
        self.volume_pattern = volume_pattern
        self.label_pattern = label_pattern
        self.preprocessing_pipeline = preprocessing_pipeline
        self.augmentation_pipeline = augmentation_pipeline
        self.postprocessing_pipeline = postprocessing_pipeline
        self.specific_pipeline = specific_pipeline
        self.ca_type = ca_type
        self.labels_names = labels_names
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        assert len(data_dir) == len(volume_pattern) == len(
            label_pattern) == len(ca_type) == len(specific_pipeline) == len(
                labels_names)

        # Fix for deepspeed
        self.setup()

    def _load_subject(self,
                      mri,
                      label,
                      labels_names,
                      ca_type,
                      dataset,
                      specific_pipeline=None):
        # TODO: Add Contrast

        subject = tio.Subject(mri=tio.ScalarImage(mri),
                              label=tio.LabelMap(label),
                              labels_names=labels_names,
                              ca_type=ca_type,
                              from_dataset=dataset)

        if specific_pipeline:
            subject = specific_pipeline(subject)

        if len(subject.label.data.unique()) == len(labels_names) + 1:
            return subject

    def _get_subject_list(self,
                          data_dir: str,
                          dataset: str,
                          volume_pattern: str,
                          label_pattern: str,
                          labels_names: list,
                          ca_type: str,
                          specific_pipeline=None):
        path = Path(data_dir)

        mris = list(path.glob(volume_pattern))
        labels = [list(mri.parent.glob(label_pattern))[0] for mri in mris]

        subjects = [
            self._load_subject(mris, labels, labels_names, ca_type, dataset,
                               specific_pipeline)
            for mris, labels in zip(mris, labels)
        ]

        return list(filter(None, subjects))

    def setup(self, stage: Optional[str] = None):
        datasets = zip(self.data_dir, self.volume_pattern, self.label_pattern,
                       self.labels_names, self.ca_type, self.datasets,
                       self.specific_pipeline)

        subjects_list = []
        for data_dir, volume_pattern, label_pattern, labels_names, ca_type, dataset, specific_pipeline in datasets:
            subjects = self._get_subject_list(data_dir, dataset, volume_pattern,
                                              label_pattern, labels_names,
                                              ca_type, specific_pipeline)
            subjects_list.extend(subjects)

        idx = list(range(len(subjects_list)))

        if self.train_val_test_idx:
            train_idx, val_idx, test_idx = self.train_val_test_idx

            if self.replace:
                train_idx = random.choices(train_idx, k=len(train_idx))

        else:
            train_idx, val_idx = train_test_split(idx,
                                                  self.train_ratio,
                                                  replace=self.replace,
                                                  k_sample=self.k_sample)
            test_idx = val_idx  # ! WARNING: ONLY FOR TESTING

        self.subjects_train_list = [subjects_list[i] for i in train_idx]
        self.subjects_val_list = [subjects_list[i] for i in val_idx]
        self.subjects_test_list = [subjects_list[i] for i in test_idx]

    def train_dataloader(self):
        transforms = [self.preprocessing_pipeline, self.augmentation_pipeline]
        if self.postprocessing_pipeline:
            transforms.append(self.postprocessing_pipeline)
        transform = tio.Compose(transforms)

        train_dataset = tio.SubjectsDataset(self.subjects_train_list,
                                            transform=transform)

        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        transforms = [self.preprocessing_pipeline]
        if self.postprocessing_pipeline:
            transforms.append(self.postprocessing_pipeline)
        transform = tio.Compose(transforms)

        val_dataset = tio.SubjectsDataset(self.subjects_val_list,
                                          transform=transform)

        return DataLoader(val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        transforms = [self.preprocessing_pipeline]
        if self.postprocessing_pipeline:
            transforms.append(self.postprocessing_pipeline)
        transform = tio.Compose(transforms)

        test_dataset = tio.SubjectsDataset(self.subjects_test_list,
                                           transform=transform)

        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


def test(path):
    # magdeburg
    # path = ["/home/cp264607/Datasets/hippocampus_magdeburg_7T/"]
    dm = MRIDataModule(data_dir=path,
                       preprocessing_pipeline=tio.Compose([
                           tio.ToCanonical(),
                           tio.ZNormalization(),
                           tio.EnsureShapeMultiple(8)
                       ]),
                       augmentation_pipeline=tio.OneOf(
                           {
                               tio.RandomAffine(): .8,
                           },
                           p=1,
                       ),
                       postprocessing_pipeline=None,
                       specific_pipeline=[
                           tio.RemapLabels({
                               1: 2,
                               2: 3,
                               3: 1,
                               4: 4,
                               5: 6,
                               6: 0,
                               7: 0,
                               8: 5,
                               9: 0,
                               10: 0,
                               11: 0,
                               12: 0,
                               13: 0,
                               17: 0
                           })
                       ],
                       batch_size=1,
                       volume_pattern=["**/tse*right*.nii.gz"],
                       label_pattern=["seg*hippocampus_right*.nii.gz"],
                       ca_type=["1/2/3"],
                       num_workers=8)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    print("keys:", batch.keys())
    print("labels:", batch["label"]["data"].unique())


# test(path="/home/cp264607/Datasets/hippocampus_magdeburg_7T/")
