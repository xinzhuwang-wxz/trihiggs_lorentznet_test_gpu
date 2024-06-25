import os
import glob
import torch
import numpy as np
from make_dataset import makeDataset 

def init_dataset(datadir, shuffle=True):
    splits = ['train', 'test', 'valid']
    patterns = {'train': 'train', 'test': 'test', 'valid': 'valid'}
    
    files = glob.glob(os.path.join(datadir, '*.root'))
    datafiles = {split: [] for split in splits}
    
    for file in files:
        for split, pattern in patterns.items():
            if pattern in file:
                datafiles[split].append(file)

    datasets = {split: makeDataset(datafiles[split][0], "HHHNtuple", shuffle=shuffle) if datafiles[split] else None for split in splits}
    
    torch_datasets = {}
    for split in splits:
        if datasets[split] is not None:
            torch_datasets[split] = []
            for i in range(len(datasets[split])):
                event_data = datasets[split][i]
                torch_event_data = {key: torch.tensor(val) if isinstance(val, (np.ndarray, list)) else torch.tensor(val) for key, val in event_data.items()}
                torch_datasets[split].append(torch_event_data)

    return torch_datasets

if __name__ == "__main__":
    datadir = "data/raw_data"
    datasets = init_dataset(datadir)

    for split, dataset in datasets.items():
        if dataset is not None:
            print(f"{split} dataset size: {len(dataset)}")
    
    for split, dataset in datasets.items():
        if dataset is not None:
            print(f"First event in {split} dataset:")
            first_event = dataset[0]
            for key, value in first_event.items():
                print(f"{key}: {value}")
                print(type(value))
            break



