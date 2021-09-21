#%%
from pathlib import Path

import ray
import torch
import pytorch_lightning as pl
from numpy.random import default_rng
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import metrics
from convex_hull_dataset import get_ch_dl
from slot_attention import SASet2Hypergraph
from set_transformer import STSet2Hypergraph

# Dataset
D_FEATS = 3
N_POINTS = torch.arange(30,31)
UNIT_NORM = True

# Model hyperparameter
D_HID = 128
TYPE = "slot_attention"  # "set_transformer"

# Training hyperparameter
BATCH_SIZE = 64
LR = 0.0003
N_EPOCHS = 4000

# Miscellaneous
SEED = 123456
RNG = default_rng(SEED)
pl.seed_everything(SEED)
N_RAY = 10
if N_RAY > 0:
    ray.init(num_cpus=N_RAY,include_dashboard=False)


class IRModel(pl.LightningModule):
    def __init__(self, max_edges):
        super().__init__()
        if TYPE == "slot_attention":
            self.net = SASet2Hypergraph(max_edges, D_FEATS, D_HID, 3)
        if TYPE == "set_transformer":
            self.net = STSet2Hypergraph(max_edges, D_FEATS, D_HID)
        self.automatic_optimization = False

    def forward(self, inputs):
        pred = self.net(inputs)
        return pred

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        bs = inputs.size(0)

        opt = self.optimizers()
        opt.zero_grad()

        pred = self.net(inputs)
        loss = metrics.LAP_loss(pred, target, n=min(N_RAY, bs)).mean(0)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            logs = {
                "loss": loss,
                "mae":  metrics.mae_cardinality(pred, target)
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return logs

    def eval_step(self, batch, batch_idx):
        inputs, target = batch
        pred = self(inputs)
        loss = metrics.LAP_loss(pred, target, n=min(N_RAY, inputs.size(0)))
        logs = {
            "loss": loss.mean(0),
            "f1": metrics.f1_score(target, pred, type="ind", d_feats=D_FEATS).mean(0),
            "precision": metrics.precision(target, pred, type="ind", d_feats=D_FEATS).mean(0),
            "recall": metrics.recall(target, pred, type="ind", d_feats=D_FEATS).mean(0),
            "mae": metrics.mae_cardinality(pred, target)
        }
        return logs

    def validation_step(self, batch, batch_idx):
        logs = self.eval_step(batch, batch_idx)
        self.log_dict({f"{k}/val":v for k,v in logs.items()})
        return logs

    def test_step(self, batch, batch_idx):
        logs = self.eval_step(batch, batch_idx)
        self.log_dict({f"{k}/test":v for k,v in logs.items()})
        return logs
        
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(parameters, lr=LR)
        return optimizer
#%%
trainloader = get_ch_dl("train", BATCH_SIZE, N_POINTS, D_FEATS, unit_norm=UNIT_NORM, add_indicator=True)
print(f"Train Dataset with max_facets={trainloader.dataset.max_facets}")
valloader = get_ch_dl("validation", BATCH_SIZE, N_POINTS, D_FEATS, unit_norm=UNIT_NORM, add_indicator=True)
print(f"Val Dataset with max_facets={valloader.dataset.max_facets}")
model = IRModel(trainloader.dataset.max_facets)
# %%
data_type = "spherical" if UNIT_NORM else "normal"
wandb.init(
    settings=wandb.Settings(start_method='spawn'),
    name=f"{TYPE} P{N_POINTS[0]}to{N_POINTS[-1]} S{SEED}",
    project=f"log_convex_hull_{data_type}",
    reinit=False,
)
logger = WandbLogger(
    log_model=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor='f1/val',
    mode='max',
)
trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    gpus=1,
    logger=logger,
    callbacks=[checkpoint_callback])
#%%
trainer.fit(model, trainloader, valloader)
# %%
testloader = get_ch_dl("test", BATCH_SIZE, N_POINTS, D_FEATS, unit_norm=UNIT_NORM, add_indicator=True)
print(f"Test Dataset with max_facets={testloader.dataset.max_facets}")
trainer.test(test_dataloaders=testloader)