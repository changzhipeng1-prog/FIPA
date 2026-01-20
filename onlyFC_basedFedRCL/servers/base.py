#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List
import torch


from servers.build import SERVER_REGISTRY
import torch

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, eigen_payloads=None):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum


    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, eigen_payloads=None):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        if self.args.server.momentum>0:
            # print("self.args.server.get('FedACG'): ",self.args.server.get('FedACG'))

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, eigen_payloads=None):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
        return model_dict


@SERVER_REGISTRY.register()
class ServerQRFC(Server):
    """Aggregate FC layer using client-provided eigenpairs; other layers are frozen (not updated)."""

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, eigen_payloads=None):
        # Without eigen info, fall back to vanilla
        if not eigen_payloads:
            return super().aggregate(local_weights, local_deltas, client_ids, model_dict, current_lr, eigen_payloads)

        C = len(client_ids)

        # Collect FC deltas and eigenpairs
        eigen_list = []
        for cid in client_ids:
            payload = eigen_payloads.get(cid) if eigen_payloads else None
            if payload is None:
                continue
            evals = payload.get("evals")
            evecs = payload.get("evecs")
            delta = payload.get("fc_delta")
            if evals is None or evecs is None:
                continue
            if delta:
                # bias first to mirror JAX params_to_subvector
                delta_vec = torch.cat([
                    delta.get("fc.bias", torch.tensor([])),
                    delta.get("fc.weight", torch.tensor([])),
                ])
            else:
                delta_vec = None
            eigen_list.append({"evals": evals, "evecs": evecs, "delta": delta_vec})

        if len(eigen_list) == 0:
            return super().aggregate(local_weights, local_deltas, client_ids, model_dict, current_lr, eigen_payloads)

        V_stack = torch.cat([item["evecs"] for item in eigen_list], dim=1)
        Sig_w = torch.cat([item["evals"] for item in eigen_list]) / C
        Q, R = torch.linalg.qr(V_stack, mode='reduced')
        K = R @ torch.diag(Sig_w) @ R.T
        K = 0.5 * (K + K.T)

        b = torch.zeros_like(Q[:, 0])
        for item in eigen_list:
            evals = item["evals"]
            evecs = item["evecs"]
            delta = item["delta"]
            if delta is None:
                continue
            b = b + (evecs @ (evals * (evecs.T @ delta))) / C

        if Q.numel() > 0:
            proj = Q.T @ b
            tau = getattr(self.args.server, 'beta', 0.0)
            gamma = getattr(self.args.server, 'gamma', 1.0)
            z = torch.linalg.solve(K + tau * torch.eye(K.shape[0]), proj)
            d_sub = gamma * (tau * (Q @ z) + (b - Q @ proj))
        else:
            d_sub = b

        # Update FC params only (keep other layers unchanged from model_dict)
        b_numel = model_dict["fc.bias"].numel()
        model_dict["fc.bias"] = model_dict["fc.bias"] + d_sub[:b_numel].view_as(model_dict["fc.bias"])
        model_dict["fc.weight"] = model_dict["fc.weight"] + d_sub[b_numel:].view_as(model_dict["fc.weight"])

        return model_dict
