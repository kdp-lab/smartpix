#! /usr/bin/env python

'''
Author: Anthony Badea
Date: September 26, 2023
'''

# python packages
import os
import datetime
import json
import argparse
import torch
import h5py
import glob
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# custom code
from batcher import loadDataFromH5
from model import ModelLightning
from convertEventFileToH5 import convertEventFileToH5

if __name__ == "__main__":

    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./minimal_config.json")
    parser.add_argument("-i",  "--inFileName", default=None, help="Input data file")
    parser.add_argument("-o", "--outDir", help="File name of the output directory", default="./checkpoints")
    parser.add_argument("-e", "--max_epochs", help="Max number of epochs to train on", default=None, type=int)
    parser.add_argument("-s", "--max_steps", help="Max number of steps to train on", default=-1, type=int)
    parser.add_argument("-d", "--device", help="Device to use.", default=None)
    parser.add_argument("-w", "--weights", help="Initial weights.", default=None)
    ops = parser.parse_args()

    # load configuration
    print(f"Using configuration file: {ops.config_file}")
    with open(ops.config_file, 'r') as fp:
        config = json.load(fp)
    print(config)

    config["model"]["weights"] = ops.weights

    # decide on device
    device = ops.device
    if not device:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "gpu")

    # load and split
    x, y = loadDataFromH5(ops.inFileName)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25)
    print(f"x_train {x_train.shape}, x_val {x_val.shape}, y_train {y_train.shape}, y_val {y_val.shape}")
    train_dataloader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, num_workers=4, pin_memory=pin_memory, batch_size=config["batch_size"]) # if multiple inputs beside just X then use DataLoader(TensorDataset(X, ...), ...)
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, num_workers=4, pin_memory=pin_memory, batch_size=config["batch_size"])
    
    # make checkpoint dir
    checkpoint_dir = os.path.join(ops.outDir, f'training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}')
    print(f"Saving checkpoints to: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create model
    model = ModelLightning(**config["model"])
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # model = torch.compile(model) # ready for pytorch 2.0 once it's more stable

    # callbacks
    callbacks = [
        ModelCheckpoint(monitor="train_loss", dirpath=checkpoint_dir, filename='cp-{epoch:04d}-{step}', every_n_train_steps = 1, save_top_k=20), # 0=no models, -1=all models, N=n models, set save_top_k=-1 to save all checkpoints
    ]

    # torch lightning trainer
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=ops.max_epochs,
        max_steps=ops.max_steps,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        # detect_anomaly=True,
        **config["trainer"]
    )
    
    # fit
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # save model
    trainer.save_checkpoint(os.path.join(checkpoint_dir,"finalWeights.ckpt"))

    # save the logged metrics
    event_file_path = glob.glob(os.path.join(checkpoint_dir, "*/*/*events.out*"))[0]
    hdf5_file_path = os.path.join(checkpoint_dir, "logged_metrics.h5")
    convertEventFileToH5(event_file_path, hdf5_file_path)
