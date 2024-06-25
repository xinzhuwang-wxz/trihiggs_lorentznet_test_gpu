import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collate import collate_fn
from init_dataset import init_dataset

def retrieve_dataloaders(batch_size, num_workers=1, datadir='./data'):
    """
    Initialize dataloaders for distributed training.

    Parameters:
    - batch_size: int, the size of the batches
    - num_workers: int, number of worker processes for data loading
    - datadir: str, directory containing the data

    Returns:
    - train_sampler: DistributedSampler, sampler for training data
    - dataloaders: dict, dataloaders for train, test, and valid datasets
    """

    # Initialize datasets
    datasets = init_dataset(datadir, shuffle=True)

    # Define a collate function with additional arguments
    collate = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1)

    # Distributed sampler for training dataset
    train_sampler = DistributedSampler(datasets['train'])

    # Construct PyTorch dataloaders for each dataset split
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size if (split == 'train') else batch_size, # prevent CUDA memory exceeded
            #shuffle=False if (split == 'train') else True,  # Shuffle only for validation and test datasets
            sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True if (split == 'train') else False,
            num_workers=num_workers,
            collate_fn=collate
        )
        for split, dataset in datasets.items()
    }

    return train_sampler, dataloaders

if __name__ == "__main__":
    batch_size = 32
    num_workers = 1
    datadir = "data/raw_data"

    # Retrieve dataloaders
    train_sampler, dataloaders = retrieve_dataloaders(batch_size, num_workers, datadir=datadir)

    print("Keys in dataloaders:", dataloaders.keys())

