#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import gc

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY
from QRMIX.base import qr_subspace_iteration_fc
import torch


@CLIENT_REGISTRY.register()
class Client():

    def __init__(self, args, client_index, model=None, loader=None):
        self.args = args
        self.client_index = client_index
        # self.loader = loader  
        self.model = model
        self.global_model =  model
        self.criterion = nn.CrossEntropyLoss()
        # eigen computation is gated by qrmix.start_round; no separate eigen flag
        return

    def setup(self, state_dict, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):

        self._update_model(state_dict)
        self.device = device

        if self.args.dataset.num_instances > 0:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)   
        else:
            train_sampler = None
        self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                   num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
                                   weight_decay=self.args.optimizer.wd)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
    
        self.trainer = trainer
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        if global_epoch == 0:
            logger.info(f"Class counts : {self.class_counts}")
        

    def _update_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')

    def get_weights(self, epoch=None):

        weights = {
            "cls": 1
        }
        
        return weights

    def _collect_full_batch(self):
        """Collect the entire local dataset into one tensor batch (CPU then moved to device)."""
        xs, ys = [], []
        for images, labels in self.loader:
            xs.append(images)
            ys.append(labels)
        x_all = torch.cat(xs, dim=0).to(self.device)
        y_all = torch.cat(ys, dim=0).to(self.device)
        return x_all, y_all

    def _extract_fc_delta_flat(self, global_state_dict, local_state_dict):
        """Flatten fc.weight/fc.bias delta for server-side FC aggregation."""
        delta = {}
        for name in ["fc.weight", "fc.bias"]:
            if name in local_state_dict:
                lw = local_state_dict[name]
                gw = global_state_dict[name].to(lw.device)
                # compute on the same device, then move to cpu for payload
                delta[name] = (lw - gw).flatten().cpu()
        return delta

    def _collect_full_batch(self):
        """Collect the entire local dataset into one tensor batch (CPU then moved to device)."""
        xs, ys = [], []
        for images, labels in self.loader:
            xs.append(images)
            ys.append(labels)
        x_all = torch.cat(xs, dim=0).to(self.device)
        y_all = torch.cat(ys, dim=0).to(self.device)
        return x_all, y_all

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        self.model.to(self.device)
        scaler = GradScaler('cuda')
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.1f')

        # logger.info(f"[Client {self.client_index}] Local training start")

        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)

        eigen_payload = None
        eigen_time = 0.0
        # Only compute eigenpairs in stage2 (QR-Mix phase); skip in stage1
        do_eigen = bool(self.args.get('qrmix') and self.args.qrmix.get('stage2'))
        # store global (broadcast) state for delta computation after local train
        broadcast_state = kwargs.get('global_state_dict', self.global_model.state_dict())
        if do_eigen:
            try:
                eigen_t0 = time.time()
                x_all, y_all = self._collect_full_batch()
                qr_cfg = self.args.get('qrmix', {}) or {}
                k = qr_cfg.get('rank_k', 10) if hasattr(qr_cfg, 'get') else qr_cfg.rank_k
                s = qr_cfg.get('oversampling', 10) if hasattr(qr_cfg, 'get') else qr_cfg.oversampling
                q = qr_cfg.get('power_iters', 1) if hasattr(qr_cfg, 'get') else qr_cfg.power_iters
                # safeguard defaults
                k = k if k else 10
                s = s if s else 10
                q = q if q else 1
                evals, evecs = qr_subspace_iteration_fc(
                    model=self.model,
                    state_dict=copy.deepcopy(self.model.state_dict()),
                    x=x_all,
                    y=y_all,
                    k=k,
                    s=s,
                    q=q,
                )
                eigen_time = time.time() - eigen_t0
                top_evals = evals[:5].cpu().numpy()
                logger.info(f"[C{self.client_index}] Eigen k={k}, s={s}, q={q}, top evals {top_evals}, time {eigen_time:.2f}s")
                eigen_payload = {
                    "client_id": self.client_index,
                    "evals": evals.cpu(),
                    "evecs": evecs.cpu(),
                    "eigen_time": eigen_time,
                }
                del x_all, y_all
            except Exception as e:
                logger.warning(f"[C{self.client_index}] Eigen computation failed: {e}")

        train_t0 = time.time()
        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            for i, (images, labels) in enumerate(self.loader):
                    
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast('cuda', enabled=self.args.use_amp):
                    losses = self._algorithm(images, labels)
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    scaler.step(self.optimizer)
                    scaler.update()

                except Exception as e:
                    print(e)

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.scheduler.step()
        train_time = time.time() - train_t0
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.2f}s, TrainTime: {train_time:.2f}s, EigenTime: {eigen_time:.2f}s, Loss: {loss_meter.avg:.3f}")

        self.model.to('cpu')
        
        # Clear GPU cache to prevent memory explosion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss_dict = {
            f'loss/{self.args.dataset.name}/cls': loss_meter.avg,
        }
        gc.collect()
        
        # Compute FC delta after local training (relative to broadcast/global state)
        if eigen_payload is not None:
            eigen_payload["fc_delta"] = self._extract_fc_delta_flat(
                global_state_dict=broadcast_state,
                local_state_dict=self.model.state_dict(),
            )

        return self.model.state_dict(), loss_dict, eigen_payload
    

    def _algorithm(self, images, labels, ) -> Dict:
        losses = defaultdict(float)

        results = self.model(images)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss

        del results
        return losses
