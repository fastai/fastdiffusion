from fastai.vision.all import *

# import torch
# from torch.utils.data import DataLoader

def save_tensors(dl, fname="train_data"):
    "Export dataloader as tensors to disk"
    X = [x for x, _ in iter(dl)]
    y = [y for _, y in iter(dl)]
    
    # stack them together
    X = torch.cat(X)
    y = torch.cat(y)
    
    # save them
    torch.save({"X":X, "y":y}, fname)
    
class TensorDataset:
    def __init__(self, X, y):
        self.X, self.y = X, y
        assert len(X) == len(y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)
    
def dls_from_tensors(train_fname, valid_fname=None, bs=64, device="cpu"):
    "Create dataloaders from tensors on fnames"
    train_data = torch.load(train_fname, map_location=device) 
    train_dl = DataLoader(TensorDataset(train_data["X"], train_data["y"]), batch_size=bs, shuffle=True, 
                          drop_last=True, pin_memory=True if device == "cpu" else False, num_workers=min(8, defaults.cpus))
    if valid_fname is not None: 
        valid_data = torch.load(valid_fname, map_location=device)
        valid_dl = DataLoader(TensorDataset(valid_data["X"], valid_data["y"]), batch_size=256, shuffle=False, 
                              drop_last=True, pin_memory=True if device == "cpu" else False, num_workers=min(8, defaults.cpus))
        return train_dl, valid_dl
    return train_dl

def cycle_dl(dl, n=1):
    "Iterate the full dataloader `n` times"
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        for _ in iter(dl):
            pass
        tf = time.perf_counter() - t0
        times.append(tf)
        print(f"Time per run: {tf}")
    return sum(times)/n