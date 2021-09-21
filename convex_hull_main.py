#%%
from pathlib import Path

import ray
import torch
import pytorch_lightning as pl
from numpy.random import default_rng
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import misc
import metrics
from convex_hull_dataset import get_ch_dl
from hypergraph_refiner import IterativeRefiner  # StackedRefiner

# Dataset
D_FEATS = 3
N_POINTS = torch.arange(30,31)
UNIT_NORM = True

# Model hyperparameter
D_HID = 128

# Training hyperparameter
N_BPTT = 2
T_BPTT = 4
T_TOTAL = 16
BATCH_SIZE = 64
LR = 0.0003
N_EPOCHS = 1000

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
        self.net = IterativeRefiner(max_edges, D_FEATS, D_HID, T_TOTAL)
        self.automatic_optimization = False
        self.sampler = misc.IntegerPartitionSampler(T_TOTAL-T_BPTT*N_BPTT, N_BPTT, RNG)

    def forward(self, inputs):
        e_t, v_t, i_t = self.net.get_initial(inputs)
        pred = self.net(inputs, e_t, v_t, i_t, t_skip=T_TOTAL-1, t_bp=1)[0][-1]
        return pred

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        bs = inputs.size(0)

        opt = self.optimizers()
        opt.zero_grad()
        loss_per_upd = []
        e_t, v_t, i_t = self.net.get_initial(inputs)
        t_pre = self.sampler()

        for t in t_pre:
            preds, e_t, v_t, i_t = self.net(inputs, e_t, v_t, i_t, t_skip=t, t_bp=T_BPTT)
            loss_per_t = [metrics.LAP_loss(p, target, n=min(N_RAY, bs)).mean(0) for p in preds]
            loss = sum(loss_per_t) / T_BPTT

            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
            e_t, v_t, i_t = e_t.detach(), v_t.detach(), i_t.detach()
            loss_per_upd.append(loss.detach())

        with torch.no_grad():
            logs = {
                "loss": loss_per_t[-1],
                "mae":  metrics.mae_cardinality(preds[-1], target),
                **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)},
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return loss

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
        return logs["loss"]

    def test_step(self, batch, batch_idx):
        logs = self.eval_step(batch, batch_idx)
        self.log_dict({f"{k}/test":v for k,v in logs.items()})
        return logs["loss"]
        
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
#%%
data_type = "spherical" if UNIT_NORM else "normal"
wandb.init(
    settings=wandb.Settings(start_method='spawn'),
    name=f"RPH P{N_POINTS[0]}to{N_POINTS[-1]} S{SEED} N_BPTT{N_BPTT} T_BPTT{T_BPTT} T_TOTAL{T_TOTAL}",
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
#%%
testloader = get_ch_dl("test", BATCH_SIZE, N_POINTS, D_FEATS, unit_norm=UNIT_NORM, add_indicator=True)
print(f"Test Dataset with max_facets={testloader.dataset.max_facets}")
trainer.test(test_dataloaders=testloader)