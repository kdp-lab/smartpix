import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import h5py
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

import sys
import glob
sys.path.append("../utils")
from convertEventFileToH5 import convertEventFileToH5

def print_size_hook(module, input, output):
    print(f"{module.__class__.__name__}: Input size: {input[0].size()}, Output size: {output.size()}")

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1+64*6*10, 128)  # Adjust based on your input size # 64 * 3 * 5
        self.fc2 = nn.Linear(128, 3)  # Adjust output size based on your task

    def forward(self, x, c):
        # process the image
        x = x.view(-1, 1, 13, 21)  # Add channel dimension for grayscale image
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.shape[0], -1) #-1, 64 * 3 * 5)  # Adjust based on your input size
        # append event variables (class token)
        x = torch.cat([x,c],-1)
        # feed forward
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = x.flatten()
        return x

# Define the LightningModule
class SimpleLightningModel(LightningModule):
    def __init__(self, debug=False):
        super(SimpleLightningModel, self).__init__()
        self.model = SimpleCNN()  # Assume you have a SimpleCNN model

        # Register the hook
        if debug:
            for layer in self.model.children():
                layer.register_forward_hook(print_size_hook)

    def forward(self, x, c):
        return self.model(x, c)

    def step(self, batch, batch_idx, version):
        x, c, y = batch
        outputs = self(x, c)
        loss = self.loss(outputs, y)
        # log the loss
        for key, val in loss.items():
            self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=True, logger=True)
        return loss["loss"]

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")
    
    def loss(self, outputs, labels):
        # total loss
        l = {}
        l["mse"] = F.mse_loss(outputs, labels.float()) # F.mse_loss
        l['loss'] = sum(l.values())
        return l

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./minimal_config.json")
    parser.add_argument("-i",  "--inFileName", default=None, help="Input data file", nargs="+")
    parser.add_argument("-o", "--outDir", help="File name of the output directory", default="./checkpoints")
    parser.add_argument("-e", "--max_epochs", help="Max number of epochs to train on", default=None, type=int)
    parser.add_argument("-s", "--max_steps", help="Max number of steps to train on", default=-1, type=int)
    parser.add_argument("-d", "--device", help="Device to use.", default=None)
    parser.add_argument("-w", "--weights", help="Initial weights.", default=None)
    parser.add_argument("--debug", help="Print out the size after each layer", action="store_true")
    return parser.parse_args()

def train(model, train_dataloader, val_dataloader, checkpoint_dir="checkpoints", max_epochs=10, max_steps=-1):

    # choose device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="train_loss", dirpath=checkpoint_dir, filename='cp-{epoch:04d}-{step}', every_n_train_steps = 1, save_top_k=20), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
        EarlyStopping(monitor="val_loss_epoch", min_delta=0.001, patience=10, verbose=True, mode='min'),
    ]

    # torch lightning trainer
    trainer = Trainer(
        accelerator=device, # be careful about accelerator/device combination
        devices=1, 
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
    )

    # fit
    trainer.fit(model, train_dataloader, val_dataloader)

    # save model
    trainer.save_checkpoint(os.path.join(checkpoint_dir, "finalWeights.ckpt"))
    
    # save the logged metrics
    event_file_path = glob.glob(os.path.join(checkpoint_dir, "*/*/*events.out*"))[0]
    hdf5_file_path = os.path.join(checkpoint_dir, "logged_metrics.h5")
    convertEventFileToH5(event_file_path, hdf5_file_path)

    return trainer

def loadData(fileList):
    
    # make sure its a list of strings
    fileList = [fileList] if isinstance(fileList, str) else fileList

    # load data
    x, c, y = None, None, None
    for inFileName in fileList: 
        # load in the data
        with h5py.File(inFileName) as f:
            d = np.array(f["data"])
            l = np.array(f["labels"])
        # variables
        nx = l[:,3]
        ny = l[:,4]
        nz = l[:,5]
        # number_eh_pairs = y[:,6]
        ylocal = l[:,7]
        pT = l[:,8]
        eta = -np.log(abs(np.tan((1/2)*(np.arctan2(nz,nx))))) #theta = alpha;all negative values go to NaN w/np.log; abs value prevents this # cotAlpha = nx/nz
        phi = (np.arctan2(nz,ny)) #to degrees *(180/torch.pi) # phi=beta # cotBeta = ny/nz
        y_temp = np.stack([eta,phi,pT],axis=-1)

        x_temp = d[:,-1]
        c_temp = np.stack([ylocal], axis=-1)

        if isinstance(x, np.ndarray):
            x = np.concatenate([x,x_temp],0)
            c = np.concatenate([c,c_temp],0)
            y = np.concatenate([y,y_temp],0)
        else:
            x = x_temp
            c = c_temp
            y = y_temp
    print(x.shape, c.shape, y.shape)
    x = torch.Tensor(x)
    c = torch.Tensor(c)
    y = torch.Tensor(y)
    return x,c,y

if __name__ == "__main__":
    
    # options
    ops = options()
    
    # load data
    x,c,y = loadData(ops.inFileName)
    x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(x, c, y, test_size = 0.25)
    train_dataloader = DataLoader(TensorDataset(x_train, c_train, y_train), shuffle=True, num_workers=4, batch_size=256)
    val_dataloader = DataLoader(TensorDataset(x_val, c_val, y_val), shuffle=False, num_workers=4, batch_size=256)
    
    # make checkpoint dir
    checkpoint_dir = os.path.join(ops.outDir, f'training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}')
    print(f"Saving checkpoints to: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create an instance of the LightningModule
    model = SimpleLightningModel(ops.debug)
        
    # train model
    trainer = train(model, train_dataloader, val_dataloader, checkpoint_dir, ops.max_epochs, ops.max_steps)

    # Set the model to evaluation mode
    model.eval()
    x,c,y = loadData(ops.inFileName[0])
    with torch.no_grad():
        outputs = model(x,c).cpu().numpy()
    labels = y.cpu().numpy()
    # Save evaluation results to an HDF5 file
    outFileName = os.path.join(checkpoint_dir, 'evaluation_results.h5')
    with h5py.File(outFileName, 'w') as f:
        f.create_dataset('outputs', data=outputs)
        f.create_dataset('labels', data=labels)

    print('Evaluation and saving to HDF5 finished!')
