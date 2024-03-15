import numpy as np
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, rank=0):
    data_loader = CustomDatasetDataLoader(opt, rank=rank)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, opt, rank=0):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        self.sampler = None
        print("rank %d %s dataset [%s] was created" % (rank, self.dataset.name, type(self.dataset).__name__))
        if opt.use_ddp and opt.isTrain:
            world_size = opt.world_size
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                    self.dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=not opt.serial_batches
                )
            self.dataloader = torch.utils.data.DataLoader(
                        self.dataset,
                        sampler=self.sampler,
                        num_workers=int(opt.num_threads / world_size), 
                        batch_size=int(opt.batch_size / world_size), 
                        drop_last=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=(not opt.serial_batches) and opt.isTrain,
                num_workers=int(opt.num_threads),
                drop_last=True
            )

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
