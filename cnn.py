import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 5, 128)  # Adjust based on your input size
        self.fc2 = nn.Linear(128, 1)  # Adjust output size based on your task

    def forward(self, x):
        x = x.view(-1, 1, 13, 21)  # Add channel dimension for grayscale image
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 64 * 3 * 5)  # Adjust based on your input size
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.flatten()
        return x

# Define the LightningModule
class SimpleLightningModel(LightningModule):
    def __init__(self):
        super(SimpleLightningModel, self).__init__()
        self.model = SimpleCNN()  # Assume you have a SimpleCNN model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels.float())
        self.log(f"loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # This method specifies how a single batch should be processed during testing

        inputs, labels = batch
        outputs = self(inputs)

        # Compute the loss (or any metric you are interested in)
        loss = self.loss(outputs, labels)
        return {'loss': loss, 'outputs': outputs, 'labels': labels}
    
    def loss(self, outputs, labels):
        return F.mse_loss(outputs, labels.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./minimal_config.json")
    parser.add_argument("-i",  "--inFileName", default=None, help="Input data file")
    parser.add_argument("-o", "--outDir", help="File name of the output directory", default="./checkpoints")
    parser.add_argument("-e", "--max_epochs", help="Max number of epochs to train on", default=None, type=int)
    parser.add_argument("-s", "--max_steps", help="Max number of steps to train on", default=-1, type=int)
    parser.add_argument("-d", "--device", help="Device to use.", default=None)
    parser.add_argument("-w", "--weights", help="Initial weights.", default=None)
    return parser.parse_args()

def train(model, train_dataloader, val_dataloader, checkpoint_dir="checkpoints", max_epochs=10, max_steps=-1):

    # choose device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="train_loss", dirpath=checkpoint_dir, filename='cp-{epoch:04d}-{step}', every_n_train_steps = 1, save_top_k=20), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
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
    trainer.save_checkpoint(os.path.join(checkpoint_dir,"finalWeights.ckpt"))
    
    return trainer

if __name__ == "__main__":
    
    # options
    ops = options()
    
    # load in the data
    with h5py.File(ops.inFileName) as f:
        x = np.array(f["data"])
        y = np.array(f["labels"])

    # the labels are x-entry, y-entry, z-entry, n_x, n_y, n_z, number_eh_pairs, y-local, pt
    # taken from https://zenodo.org/record/7331128
    # other relevant parameters to compute are https://github.com/kdipetri/semiparametric/blob/master/processing/datagen.py#L110C9-L115C95
    # cotAlpha = y[:,3]/y[:,5] # n_x/n_z
    cotBeta = y[:,4]/y[:,5] # n_y/n_z
    # cotBeta = abs(cotBeta)
    # sensor_thickness = 100 #um
    # x_midplane = y[:,0] + cotBeta*(sensor_thickness/2 - y[:,2]) # x-entry + cotAlpha*(sensor_thickness/2 - z-entry)
    # y_midplane = y[:,1] + cotBeta*(sensor_thickness/2 - y[:,2]) # y-entry + cotBeta*(sensor_thickness/2 - z-entry)
    y_local = y[:,7]

    x = x[:,-1]
    # form final input
    # x = np.concatenate([x, y_local.reshape(-1,1)],-1)

    # event selection
    # mask = abs(y[:,8]) >= 0.3 # pt>0.3 GeV
    # print(mask.sum()/mask.shape[0])

    # convert to tensor
    x = torch.Tensor(x) #[mask])
    y = torch.Tensor(cotBeta) #[mask])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25)
    train_dataloader = DataLoader(TensorDataset(x, y), shuffle=True, num_workers=4, batch_size=256)
    val_dataloader = DataLoader(TensorDataset(x, y), shuffle=True, num_workers=4, batch_size=256)

    # Create an instance of the LightningModule
    model = SimpleLightningModel()
    trainer = train(model, train_dataloader, val_dataloader, ops.outDir, max_epochs=ops.max_epochs, max_steps=ops.max_steps)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(x).cpu().numpy()
    labels = y.cpu().numpy()
    # Save evaluation results to an HDF5 file
    with h5py.File('evaluation_results.h5', 'w') as f:
        f.create_dataset('outputs', data=outputs)
        f.create_dataset('labels', data=labels)

    print('Evaluation and saving to HDF5 finished!')
