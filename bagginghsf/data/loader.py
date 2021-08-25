from functools import partial

import torchio as tio
from omegaconf.dictconfig import DictConfig

from ..data.dataloader import MRIDataModule


def get_subdatasets(dataset):
    subdatasets = dataset.patterns.keys()
    mris = [dataset.patterns[s]["mri"] for s in subdatasets]
    labels = [dataset.patterns[s]["label"] for s in subdatasets]

    return list(subdatasets), mris, labels


def load_from_config(data_cfg: DictConfig):
    datasets = data_cfg.datasets.keys()
    paths = [
        data_cfg.main_path + data_cfg.datasets[d]['path'] for d in datasets
    ]

    resulting_datasets = []
    resulting_paths = []
    volume_pattern = []
    label_pattern = []
    labels_names = []
    ca_type = []
    specific_pipeline = []
    for d, p in zip(datasets, paths):
        subdatasets, mris, labels = get_subdatasets(data_cfg.datasets[d])
        subpaths = [p] * len(subdatasets)
        resulting_datasets.extend([d + "_" + s for s in subdatasets])
        resulting_paths.extend(subpaths)
        volume_pattern.extend(mris)
        label_pattern.extend(labels)
        labels_names.extend([data_cfg.datasets[d]["labels_names"]] *
                            len(subdatasets))
        ca_type.extend([data_cfg.datasets[d]["ca_type"]] * len(subdatasets))

        pipeline = [tio.RemapLabels(data_cfg.datasets[d]["labels"])
                   ] * len(subdatasets)

        specific_pipeline.extend(pipeline)

    assert len(resulting_datasets) == len(resulting_paths) == len(
        volume_pattern) == len(label_pattern) == len(ca_type) == len(
            specific_pipeline) == len(labels_names)

    return partial(MRIDataModule,
                   data_dir=resulting_paths,
                   datasets=resulting_datasets,
                   specific_pipeline=specific_pipeline,
                   batch_size=data_cfg.batch_size,
                   train_ratio=data_cfg.train_ratio,
                   k_sample=data_cfg.k_sample,
                   replace=data_cfg.replace,
                   volume_pattern=volume_pattern,
                   label_pattern=label_pattern,
                   labels_names=labels_names,
                   ca_type=ca_type,
                   num_workers=data_cfg.num_workers)
