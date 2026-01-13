import torch
from image_datasets import create_dataset
from mmreg import get_dataset_and_topo_repr, TopoAlgoType

print('Creating dataset...')
dataset = create_dataset('data/cifar10/', (32, 32), 128, 3)
print(f'Dataset created, length: {len(dataset)}')

print('Computing TDA representation...')
topo_algo = TopoAlgoType.PCA
dtype = torch.float32
dataset_with_topo = get_dataset_and_topo_repr(dataset, dtype, topo_algo, 100)
print('TDA computation complete')
print(f'Dataset type: {type(dataset_with_topo)}')
print(f'Dataset length: {len(dataset_with_topo)}')

print('\nTesting first batch...')
from torch.utils.data import DataLoader
loader = DataLoader(dataset_with_topo, batch_size=128)
batch_iter = iter(loader)
inputs, data_repr = next(batch_iter)
print(f'Input shape: {inputs.shape}')
print(f'Data repr shape: {data_repr.shape}')
print('Test successful!')
