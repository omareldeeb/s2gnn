import os
import logging


import numpy as np
import yaml
import string
import ast
import random
import time
from datetime import datetime
import torch.nn as nn

from graphgps.loader.gemnet import gemnet_device
from graphgps.loader.gemnet.data_container import DataContainer
from graphgps.loader.gemnet.data_provider import DataProvider
from graphgps.loader.gemnet.metrics import BestMetrics, Metrics
from graphgps.loader.gemnet.trainer import Trainer
from graphgps.network.gemnet import GemNet
# from torch.utils.tensorboard import SummaryWriter
# from graphgps.loader.dataset.wrapped_md17 import MD17
# from torch.utils.data import DataLoader

from torch_geometric.data import DataLoader
from torch_geometric.datasets import MD17

from torch_geometric.transforms import Compose, RadiusGraph, Distance
from tqdm import tqdm
# from training.trainer import Trainer
# from training.metrics import Metrics, BestMetrics
# from training.data_container import DataContainer
# from gemnet.training.data_provider import DataProvider

import torch



with open('gemnet_config.yml', 'r') as c:
    config = yaml.safe_load(c)
    
# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass


num_spherical = config["num_spherical"]
num_radial = config["num_radial"]
num_blocks = config["num_blocks"]
emb_size_atom = config["emb_size_atom"]
emb_size_edge = config["emb_size_edge"]
emb_size_trip = config["emb_size_trip"]
emb_size_quad = config["emb_size_quad"]
emb_size_rbf = config["emb_size_rbf"]
emb_size_cbf = config["emb_size_cbf"]
emb_size_sbf = config["emb_size_sbf"]
num_before_skip = config["num_before_skip"]
num_after_skip = config["num_after_skip"]
num_concat = config["num_concat"]
num_atom = config["num_atom"]
emb_size_bil_quad = config["emb_size_bil_quad"]
emb_size_bil_trip = config["emb_size_bil_trip"]
triplets_only = config["triplets_only"]
forces_coupled = config["forces_coupled"]
direct_forces = config["direct_forces"]
mve = config["mve"]
cutoff = config["cutoff"]
int_cutoff = config["int_cutoff"]
envelope_exponent = config["envelope_exponent"]
extensive = config["extensive"]
output_init = config["output_init"]
scale_file = config["scale_file"]
data_seed = config["data_seed"]
dataset = config["dataset"]
val_dataset = config["val_dataset"]
num_train = config["num_train"]
num_val = config["num_val"]
logdir = config["logdir"]
loss = config["loss"]
tfseed = config["tfseed"]
num_steps = config["num_steps"]
rho_force = float(config["rho_force"])
ema_decay = config["ema_decay"]
weight_decay = config["weight_decay"]
grad_clip_max = config["grad_clip_max"]
agc = config["agc"]
decay_patience = config["decay_patience"]
decay_factor = config["decay_factor"]
decay_cooldown = config["decay_cooldown"]
batch_size = config["batch_size"]
evaluation_interval = config["evaluation_interval"]
patience = config["patience"]
save_interval = config["save_interval"]
learning_rate = config["learning_rate"]
warmup_steps = config["warmup_steps"]
decay_steps = config["decay_steps"]
decay_rate = config["decay_rate"]
staircase = config["staircase"]
restart = config["restart"]
comment = config["comment"]





data_container = DataContainer(
    path=dataset,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    triplets_only=triplets_only,
    # transforms=None,
    # addID=False,
)

if num_train == 0:
    num_train = len(data_container)

if num_val == 0:
    num_val = len(data_container)// 10

length = len(data_container)
print(length)


logging.info(f"Training data size: {num_train}")
logging.info(f"Validation data size: {num_val}")




data_provider = DataProvider(
    data_container,
    num_train,
    num_val,
    batch_size,
    seed=data_seed,
    shuffle=True,
    random_split=True,
)
val_data_provider = data_provider



train = {}
validation = {}
train["dataset_iter"] = data_provider.get_dataset("train")
validation["dataset_iter"] = val_data_provider.get_dataset("val")



def load_model():
    device = torch.device(gemnet_device.gemnet_device)
    model = GemNet(
    num_spherical=num_spherical,
    num_radial=num_radial,
    num_blocks=num_blocks,
    emb_size_atom=emb_size_atom,
    emb_size_edge=emb_size_edge,
    emb_size_trip=emb_size_trip,
    emb_size_quad=emb_size_quad,
    emb_size_rbf=emb_size_rbf,
    emb_size_cbf=emb_size_cbf,
    emb_size_sbf=emb_size_sbf,
    emb_size_bil_quad=emb_size_bil_quad,
    emb_size_bil_trip=emb_size_bil_trip,
    num_before_skip=num_before_skip,
    num_after_skip=num_after_skip,
    num_concat=num_concat,
    num_atom=num_atom,
    triplets_only=triplets_only,
    direct_forces=direct_forces,
    forces_coupled=forces_coupled,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    envelope_exponent=envelope_exponent,
    extensive=extensive,
    output_init=output_init,
    scale_file=scale_file,
    activation="swish")

    return model.to(device)

# def test(model, loader, device):
#     model.eval()
    
#     total_mae = 0.0
#     with torch.no_grad():
#         for batch in tqdm(loader, desc='Test', unit='batch'):
#             batch = batch.to(device)
#             preds_norm = model(batch)
#             preds = preds_norm
#             total_mae += (preds - batch.energy.view(-1).to(device)).abs().sum().item()
#     return total_mae / len(loader.dataset)



# def dict2device(data, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     for key in data:
#         data[key] = data[key].to(device)
#     return data

# def get_mae(self, targets, pred):
#     """
#     Mean Absolute Error
#     """
#     return torch.nn.functional.l1_loss(pred, targets, reduction="mean")

# def get_rmse(self, targets, pred):
#     """
#     Mean L2 Error
#     """
#     return torch.mean(torch.norm((pred - targets), p=2, dim=1))



# def predict(model, inputs, mve):

#     energy, forces = model(inputs)

#     if mve:
#         mean_energy = energy[:, :1]
#         var_energy = torch.nn.functional.softplus(energy[:, 1:])
#         mean_forces = forces[:, 0, :]
#         var_forces = torch.nn.functional.softplus(forces[:, 1, :])
#         return mean_energy, var_energy, mean_forces, var_forces
#     else:
#         if len(forces.shape) == 3:
#             forces = forces[:, 0]
#         return energy, None, forces, None


# def train_loop(model, loader, optimizer, device):
#     model.train()
#     criterion = nn.MSELoss()
    
#     sum_loss = 0.0
#     batch_count = 0

#     for batch in tqdm(loader, desc='Train', unit='batch'):
#         inputs, targets = batch

#         inputs, targets = dict2device(inputs), dict2device(targets)
#         mean_energy, var_energy, mean_forces, var_forces = predict(inputs)


#         optimizer.zero_grad()
#         targets = batch.energy.view(-1).to(device)
#         preds_norm = model(batch)
#         loss = criterion(preds_norm, targets)
#         loss.backward()
#         optimizer.step()
#         batch_count += 1
#         sum_loss += loss.item()
#         avg_loss = sum_loss / batch_count
#         tqdm.write(f"Batch {batch_count}: Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
#     return sum_loss / batch_count

def gemnet_train_loop(log_dir, log_path_model, log_path_training, best_path_model):
    # summary_writer = SummaryWriter(log_dir)
    steps_per_epoch = int(np.ceil(num_train / batch_size))

    for step in range(step_init + 1, num_steps + 1):

        # keep track of the learning rate
        if step % 10 == 0:
            lr = trainer.schedulers[0].get_last_lr()[0]
            # summary_writer.add_scalar("lr", lr, global_step=step)

        # Perform training step
        trainer.train_on_batch(train["dataset_iter"], train["metrics"])

        # Save progress
        if step % save_interval == 0:
            torch.save({"model": model.state_dict()}, log_path_model)
            torch.save(
                {"trainer": trainer.state_dict(), "step": step}, log_path_training
            )

        # Check performance on the validation set
        if step % evaluation_interval == 0:

            # Save backup variables and load averaged variables
            trainer.save_variable_backups()
            trainer.load_averaged_variables()

            # Compute averages
            for i in range(int(np.ceil(num_val / batch_size))):
                trainer.test_on_batch(validation["dataset_iter"], validation["metrics"])

            # Update and save best result
            if validation["metrics"].loss < metrics_best.loss:
                metrics_best.update(step, validation["metrics"])
                torch.save(model.state_dict(), best_path_model)

            # write to summary writer
            # metrics_best.write(summary_writer, step)

            epoch = step // steps_per_epoch
            train_metrics_res = train["metrics"].result(append_tag=False)
            val_metrics_res = validation["metrics"].result(append_tag=False)
            metrics_strings = [
                f"{key}: train={train_metrics_res[key]:.6f}, val={val_metrics_res[key]:.6f}"
                for key in validation["metrics"].keys
            ]
            logging.info(
                f"{step}/{num_steps} (epoch {epoch}): " + "; ".join(metrics_strings)
            )

            # decay learning rate on plateau
            trainer.decay_maybe(validation["metrics"].loss)

            # train["metrics"].write(summary_writer, step)
            # validation["metrics"].write(summary_writer, step)
            train["metrics"].reset_states()
            validation["metrics"].reset_states()

            # Restore backup variables
            trainer.restore_variable_backups()

            # early stopping
            if step - metrics_best.step > patience * evaluation_interval:
                break



if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device(gemnet_device.gemnet_device)
    
    model = load_model()
    radius = 5.0
    transform = Compose([RadiusGraph(r=radius, max_num_neighbors=32), Distance(norm=False)])
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        decay_patience=decay_patience,
        decay_factor=decay_factor,
        decay_cooldown=decay_cooldown,
        grad_clip_max=grad_clip_max,
        rho_force=rho_force,
        mve=mve,
        loss=loss,
        staircase=staircase,
        agc=agc,
    )
    train["metrics"] = Metrics("train", trainer.tracked_metrics)
    validation["metrics"] = Metrics("val", trainer.tracked_metrics)
    metrics_best = BestMetrics("./gemnet_tests", validation["metrics"])
    logging.info("Freshly initialize model")
    metrics_best.inititalize()
    step_init = 0

    gemnet_train_loop(
        './gemnet_tests',
        './gemnet_tests/model_log.txt',
        './gemnet_tests/log.txt',
        './gemnet_tests/best_model'
    )

    result = {key + "_best": val for key, val in metrics_best.items()}
    # dataset = MD17(root='data/MD17', name='aspirin', transform=transform)

    # x = dataset[0]

    # split = int(len(dataset) * 0.8)
    # train_dataset = dataset[:split]
    # test_dataset = dataset[split:]

    # Mean-normalization
    # all_train_energies = torch.cat([d.energy.view(-1) for d in train_dataset])
    # mean_energy = all_train_energies.mean().to(device)
    # print(f"Mean training energy: {mean_energy:.4f}")

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 101):
        print(f"Epoch {epoch:03d}")
        train_loss = train_loop(model, train["dataset_iter"], optimizer, device)
        test_mae = test(model, validation["dataset_iter"], device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f}")