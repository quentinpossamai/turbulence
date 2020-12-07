import torch
import zipfile
import os
# x = torch.empty(5, 3)  # Can contain nans
# x = torch.rand(5, 3)  # Random uniform(0, 1)
# x = torch.zeros(5, 3, dtype=torch.long)
# x = torch.tensor([5.5, 3])
# x = x.new_zeros(2)  # Keeps dtype unless dtype is kept.
# x = torch.randn_like(x)
# print(x.shape)
# y = torch.rand(5, 3)
os.chdir('/Users/quentin/phd/lirisvis')
with zipfile.ZipFile("sf2.zip", 'r') as f:
    # zip_ref.extractall("Store")
    for name in f.namelist():
        if name[-1] != '/':
            data = f.read(name)
            print(name, len(data), repr(data[:10]))
    pass