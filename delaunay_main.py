#%%
from pathlib import Path

import ray
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from numpy.random import default_rng
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import misc
import metrics
from graph_refiner import IterativeRefiner
from delaunay_data import get_dt_dl

# Dataset
D_FEATS = 2
N_POINTS = torch.arange(20,81)

# Model hyperparameter
D_HID = 256

# Training hyperparameter
N_BPTT = 4
T_BPTT = 4
T_TOTAL = 32
BATCH_SIZE = 64
LR = 0.0003
N_EPOCHS = 100

# Miscellaneous
SEED = 123456
RNG = default_rng(SEED)
pl.seed_everything(SEED)

class IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = IterativeRefiner(D_FEATS, D_HID, T_TOTAL)
        self.automatic_optimization = False
        self.sampler = misc.IntegerPartitionSampler(T_TOTAL-T_BPTT*N_BPTT, N_BPTT, RNG)

    def forward(self, inputs):
        v_t, i_t = self.net.get_initial(inputs)
        pred = self.net(inputs, v_t, i_t, t_skip=T_BPTT-1, t_bp=1)[0][-1]
        return pred

    def training_step(self, batch, batch_idx):
        inputs, target = batch

        opt = self.optimizers()
        opt.zero_grad()
        loss_per_upd = []
        v_t, i_t = self.net.get_initial(inputs)
        t_pre = self.sampler()

        for i, t in enumerate(t_pre):
            preds, v_t, i_t = self.net(inputs, v_t, i_t, t_skip=t, t_bp=T_BPTT)
            loss_per_t = [F.binary_cross_entropy(p, target).mean(0) for p in preds]
            loss = sum(loss_per_t) / T_BPTT

            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
            v_t, i_t = v_t.detach(), i_t.detach()
            loss_per_upd.append(loss.detach())

        with torch.no_grad():
            logs = {
                "loss": loss_per_t[-1],
                "mae":  metrics.mae_cardinality(preds[-1], target),
                **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)},
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return logs

    def eval_step(self, batch, batch_idx):
        inputs, target = batch
        pred = self(inputs)
        loss = F.binary_cross_entropy(pred, target)
        acc, f1, prec, rec = metrics.delaunay_adj_metrics(target, pred)
        logs = {
            "loss": loss.mean(0),
            "accuracy": acc.mean(0),
            "f1": f1.mean(0),
            "precision": prec.mean(0),
            "recall": rec.mean(0),
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
trainloader = get_dt_dl("train", BATCH_SIZE, N_POINTS, D_FEATS)
valloader = get_dt_dl("validation", BATCH_SIZE, N_POINTS, D_FEATS)
model = IRModel()
# %%
wandb.init(
    settings=wandb.Settings(start_method='spawn'),
    name=f"RPH P{N_POINTS[0]}to{N_POINTS[-1]} S{SEED} N_BPTT{N_BPTT} T_BPTT{T_BPTT} T_TOTAL{T_TOTAL}",
    project="log_delaunay_triangulation",
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
testloader = get_dt_dl("test", BATCH_SIZE, N_POINTS, D_FEATS)
trainer.test(test_dataloaders=testloader)