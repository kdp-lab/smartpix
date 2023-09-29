import torch
import pytorch_lightning as pl
import model_blocks 

class ModelLightning(pl.LightningModule):

    def __init__(self,
                 model_config = {},
                 lr = 1e-3,
                 weights = None):
        super().__init__()

        self.model = model_blocks.Model(**model_config)
        print(self.model)
        self.lr = lr

        # use the weights hyperparameters
        if weights: 
            ckpt = torch.load(weights,map_location=self.device)
            self.load_state_dict(ckpt["state_dict"])
        
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x
        
    def step(self, batch, batch_idx, version):
        
        # forward pass
        x, y = batch
        x = self(x)

        # compute loss
        loss = self.loss(x, y)

        # log the loss
        for key, val in loss.items():
            self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=True, logger=True)
        
        # log the accuracy
        # self.log(f"{version}_acc", self.acc(x, y), prog_bar=True, on_step=True)

        #print(loss["loss"])
        return loss["loss"]
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
        return optimizer
    
    def acc(self, yhat, y):
        return None # (torch.argmax(yhat, dim=1) == y).sum() / y.numel()

    def loss(self, yhat, y):

        ''' 
        cout/cin = [B, E]
        '''

        # print(yhat[0], y[0])

        # total loss
        l = {}
        l["mse"] = torch.mean((yhat-y)**2)

        # get total
        l['loss'] = sum(l.values())

        return l
