#%%
from pathlib import Path

import ray
import torch
import pytorch_lightning as pl
import wandb
from numpy.random import default_rng
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import misc
import metrics
import partitioning_data
import partitioning_eval
from hypergraph_refiner import IterativeRefiner

# Dataset
D_FEATS = 10
MAX_EDGES = 10

# Model hyperparameter
D_HID = 128

# Training hyperparameter
N_BPTT = 2
T_BPTT = 4
T_TOTAL = 16
BATCH_SIZE = 2048
LR = 0.0003
N_EPOCHS = 400

# Miscellaneous
SEED = 123456
RNG = default_rng(SEED)
pl.seed_everything(SEED)
N_RAY = 10
if N_RAY > 0:
    ray.init(num_cpus=N_RAY,include_dashboard=False)


class IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = IterativeRefiner(MAX_EDGES, D_FEATS, D_HID, T_TOTAL)
        self.sampler = misc.IntegerPartitionSampler(T_TOTAL-T_BPTT*N_BPTT, N_BPTT, RNG)
        self.automatic_optimization = False

    def forward(self, inputs):
        e_t, v_t, i_t = self.net.get_initial(inputs)
        pred = self.net(inputs, e_t, v_t, i_t, t_skip=T_TOTAL-1, t_bp=1)[0][-1]
        return pred

    def training_step(self, batch, batch_idx):
        inputs, _, target = batch
        target = torch.cat([target, target.bool().any(-1, keepdim=True).float()], dim=-1)
        bs = inputs.size(0)

        opt = self.optimizers()
        opt.zero_grad()
        loss_per_upd = []
        e_t, v_t, i_t = self.net.get_initial(inputs)
        t_pre = self.sampler()
        for t in t_pre:
            preds, e_t, v_t, i_t = self.net(inputs, e_t, v_t, i_t, t_skip=t, t_bp=T_BPTT)
            
            preds = torch.cat(preds, dim=0)
            targets = target.repeat(T_BPTT, 1, 1)
            loss_per_t = metrics.LAP_loss(
                preds, 
                targets, 
                n=min(N_RAY, bs))
            loss_inc = loss_per_t.mean(0)

            preds = preds.clamp(0,1)

            pred_adj = torch.bmm(preds[...,:-1].transpose(1,2), preds[...,:-1])
            target_adj = torch.bmm(targets[...,:-1].transpose(1,2), targets[...,:-1])
            f1 = metrics.f1_score(target_adj, pred_adj, type="adj")

            loss = loss_inc - f1.mean(0)

            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
            e_t, v_t, i_t = e_t.detach(), v_t.detach(), i_t.detach()
            loss_per_upd.append(loss.detach())

        with torch.no_grad():
            logs = {
                "loss": loss_per_t[-bs:].mean(0),
                "mae":  metrics.mae_cardinality(preds[-bs:], target),
                "f1": f1[-bs:].mean(0),
                **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)},
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return loss

    def eval_step(self, batch, batch_idx):
        inputs, _, target = batch
        target = torch.cat([target, target.bool().any(-1, keepdim=True).float()], dim=-1)
        pred = self(inputs)
        target_adj = torch.bmm(target[...,:-1].transpose(1,2), target[...,:-1])
        pred_adj = torch.bmm(pred[...,:-1].transpose(1,2), pred[...,:-1])
        logs = {
            "loss": metrics.LAP_loss(pred, target, n=min(N_RAY, inputs.size(0))).mean(0),
            "f1": metrics.f1_score(target_adj, pred_adj, type="adj").mean(0),
            "precision": metrics.precision(target_adj, pred_adj, type="adj").mean(0),
            "recall": metrics.recall(target_adj, pred_adj, type="adj").mean(0),
            "mae": metrics.mae_cardinality(pred, target),
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
debug_load = False
trainloader = partitioning_data.get_data_loader("train", BATCH_SIZE, debug_load=debug_load)
valloader = partitioning_data.get_data_loader("validation", BATCH_SIZE, debug_load=debug_load)
model = IRModel()
# %%
wandb.init(
    name=f"RPH S{SEED} N_BPTT{N_BPTT} T_BPTT{T_BPTT} T_TOTAL{T_TOTAL}",
    project="particle_partition",
    reinit=False,
    settings=wandb.Settings(start_method="fork"),
)
logger = WandbLogger(
    log_model=True,
)
checkpoint_callback_loss = ModelCheckpoint(
    monitor='loss/val',
    mode='min',
)
trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    gpus=1,
    logger=logger, 
    callbacks=[checkpoint_callback_loss])
#%%
trainer.fit(model, trainloader, valloader, )
#%%
testds = partitioning_data.JetGraphDataset("test", random_permutation=False)
print("Test")
model = model.cuda()
print(partitioning_eval.test_performance(model, testds))