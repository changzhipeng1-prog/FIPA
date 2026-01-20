from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type, Any
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import tqdm
import wandb
import gc

import pickle, os
import numpy as np

import logging
logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

import time, io, copy

from trainers.build import TRAINER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, DatasetSplitSubset, get_dataset
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed, save_checkpoint
from omegaconf import DictConfig,OmegaConf


from netcal.metrics import ECE
import matplotlib.pyplot as plt



@TRAINER_REGISTRY.register()
class Trainer():

    def __init__(self,
                 model: nn.Module,
                 client_type: Type,
                 server: Server,
                 evaler_type: Type,
                 datasets: Dict,
                 device: torch.device,
                 args: DictConfig,
                 multiprocessing: Dict = None,
                 **kwargs) -> None:

        self.args = args
        self.device = device
        self.model = model

        self.checkpoint_path = Path(self.args.checkpoint_path)
        mode = self.args.split.mode 
        if self.args.split.mode == 'dirichlet':
            mode += str(self.args.split.alpha)
        self.exp_path = self.checkpoint_path / self.args.dataset.name / mode / self.args.exp_name
        logger.info(f"Exp path : {self.exp_path}")

        ### training config
        trainer_args = self.args.trainer
        self.num_clients = trainer_args.num_clients
        self.participation_rate = trainer_args.participation_rate
        self.global_rounds = trainer_args.global_rounds
        self.lr = trainer_args.local_lr
        self.local_lr_decay = trainer_args.local_lr_decay


        self.clients: List[Client] = [client_type(self.args, client_index=c, model=copy.deepcopy(self.model)) for c in range(self.args.trainer.num_clients)]
        self.server = server
        if self.args.server.momentum > 0:
            self.server.set_momentum(self.model)

        self.datasets = datasets
        self.local_dataset_split_ids = get_dataset(self.args, self.datasets['train'], mode=self.args.split.mode)

        test_loader = DataLoader(self.datasets["test"],
                                batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        eval_params = {
            "test_loader": test_loader,
            "device": eval_device,
            "args": args,
        }
        self.eval_params = eval_params
        self.eval_device = eval_device
        self.evaler = evaler_type(**eval_params)
        logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")

        self.start_round = 0
        if self.args.get('load_model_path'):
            self.load_model()



    def local_update(self, device, task_queue, result_queue):
        if self.args.multiprocessing:
            torch.cuda.set_device(device)
            initalize_random_seed(self.args)

        while True:
            task = task_queue.get()
            if task is None:
                break
            client = self.clients[task['client_idx']]

            # local_dataset = DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[task['client_idx']])
            local_dataset = DatasetSplitSubset(
                self.datasets['train'],
                idxs=self.local_dataset_split_ids[task['client_idx']],
                subset_classes=self.args.dataset.get('subset_classes'),
                )

            setup_inputs = {
                'state_dict': task['state_dict'],
                'device': device,
                'local_dataset': local_dataset,
                'local_lr': task['local_lr'],
                'global_epoch': task['global_epoch'],
                'trainer': self,
            }
            client.setup(**setup_inputs)

            # Local Training (allow clients that return 2-tuple or 3-tuple)
            result = client.local_train(global_epoch=task['global_epoch'], global_state_dict=task['state_dict'])
            if isinstance(result, tuple) and len(result) == 3:
                local_model, local_loss_dict, eigen_payload = result
            else:
                local_model, local_loss_dict = result
                eigen_payload = None

            # ensure CPU tensors before sending via Manager queue
            local_model_cpu = {k: v.detach().cpu() for k, v in local_model.items()}
            safe_loss_dict = {}
            for k, v in local_loss_dict.items():
                if isinstance(v, torch.Tensor):
                    safe_loss_dict[k] = v.detach().cpu()
                else:
                    safe_loss_dict[k] = v

            if eigen_payload is not None:
                safe_eigen = {}
                for k, v in eigen_payload.items():
                    if isinstance(v, torch.Tensor):
                        safe_eigen[k] = v.detach().cpu()
                    elif isinstance(v, dict):
                        # e.g., fc_delta dict of tensors
                        safe_eigen[k] = {kk: vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
                    else:
                        safe_eigen[k] = v
                eigen_payload = safe_eigen

            result_queue.put((local_model_cpu, safe_loss_dict, eigen_payload))
            if not self.args.multiprocessing:
                break

    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            ngpus_per_node = torch.cuda.device_count()
            task_queues = [mp.Queue() for _ in range(M)]
            processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]

            # start all processes
            for p in processes:
                p.start()

        # Evaluate loaded model before training starts (if model was loaded)
        if self.args.get('load_model_path') and self.start_round > 0:
            logger.info("=" * 80)
            logger.info("Evaluating loaded model before training starts")
            logger.info(f"Loaded model from: {self.args.load_model_path}")
            logger.info(f"Model was saved at epoch: {self.start_round - 1}")
            logger.info("=" * 80)
            self.evaluate(epoch=self.start_round - 1)
            logger.info("=" * 80)
            logger.info("")

        for epoch in range(self.start_round, self.global_rounds):

            self.lr_update(epoch=epoch)

            global_state_dict = copy.deepcopy(self.model.state_dict())
            prev_model_weight = copy.deepcopy(self.model.state_dict())
            
            # Select clients
            if self.participation_rate < 1.:
                selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
            else:
                selected_client_ids = range(len(self.clients))
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            current_lr = self.lr

            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)
            local_eigens = {}

            local_models = []

            # FedACG lookahead momentum
            if self.args.server.get('FedACG'):
                assert(self.args.server.momentum > 0)
                self.model= copy.deepcopy(self.server.FedACG_lookahead(copy.deepcopy(self.model)))
                global_state_dict = copy.deepcopy(self.model.state_dict())

            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    'state_dict': self.model.state_dict(),
                    'client_idx': client_idx,
                    #'lr': current_lr,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    task_queues[i].put(task_queue_input)
                    # logger.info(f"[C{client_idx}] put queue")
                else:
                    task_queue = mp.Queue()
                    task_queue.put(task_queue_input)
                    self.local_update(self.device, task_queue, result_queue)

                    result = result_queue.get()
                    if len(result) == 3:
                        local_state_dict, local_loss_dict, eigen_payload = result
                    else:
                        local_state_dict, local_loss_dict = result
                        eigen_payload = None
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    #local_state_dict = local_model.state_dict()
                    local_models.append(local_state_dict)
                    if eigen_payload is not None:
                        local_eigens[eigen_payload["client_id"]] = eigen_payload

                    # Ensure tensors are on CPU to save GPU memory
                    for param_key in local_state_dict:
                        if isinstance(local_state_dict[param_key], torch.Tensor):
                            local_state_dict[param_key] = local_state_dict[param_key].cpu()
                        if isinstance(global_state_dict[param_key], torch.Tensor):
                            global_state_dict[param_key] = global_state_dict[param_key].cpu()
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])
                    
                    # Clear GPU cache after each client
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


            if self.args.multiprocessing:
                for _ in range(len(selected_client_ids)):
                    # Retrieve results from the queue
                    result = result_queue.get()
                    if len(result) == 3:
                        local_state_dict, local_loss_dict, eigen_payload = result
                    else:
                        local_state_dict, local_loss_dict = result
                        eigen_payload = None
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    local_models.append(local_state_dict)
                    if eigen_payload is not None:
                        local_eigens[eigen_payload["client_id"]] = eigen_payload

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    for param_key in local_state_dict:
                        # Ensure tensors are on CPU to save GPU memory
                        if isinstance(local_state_dict[param_key], torch.Tensor):
                            local_state_dict[param_key] = local_state_dict[param_key].cpu()
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")

            # Clear GPU cache after client training to prevent memory explosion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Server-side
            # Ensure global_state_dict is on CPU before aggregation
            cpu_global_state_dict = {}
            for key, value in global_state_dict.items():
                if isinstance(value, torch.Tensor):
                    cpu_global_state_dict[key] = value.cpu()
                else:
                    cpu_global_state_dict[key] = value
            
            eigen_payloads = local_eigens if len(local_eigens) > 0 else None

            updated_global_state_dict = self.server.aggregate(
                local_weights,
                local_deltas,
                selected_client_ids,
                copy.deepcopy(cpu_global_state_dict),
                current_lr,
                eigen_payloads=eigen_payloads,
            )
            self.model.load_state_dict(updated_global_state_dict)
            
            # Clear temporary variables
            del local_weights, local_deltas, local_models, cpu_global_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            local_datasets = [DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[client_id]) for client_id in selected_client_ids]

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            model_device = next(self.model.parameters()).device
            if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch, local_datasets=local_datasets)
                # Clear GPU cache after evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save checkpoints logic:
            # 1. Regular save_freq (if set and epoch matches)
            # 2. Last N epochs (if save_last_n_epochs is set and epoch is in last N)
            # 3. Final epoch (always save)
            should_save = False
            if self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0:
                should_save = True
            
            # Save last N epochs if specified (e.g., last 100 epochs: 900-1000)
            if hasattr(self.args, 'save_last_n_epochs') and self.args.save_last_n_epochs > 0:
                total_rounds = self.args.trainer.global_rounds
                start_save_epoch = total_rounds - self.args.save_last_n_epochs
                if epoch >= start_save_epoch:
                    should_save = True
            
            # Always save final epoch
            if (epoch + 1 == self.args.trainer.global_rounds):
                should_save = True
            
            if should_save:
                self.save_model(epoch=epoch)

            self.wandb_log(wandb_dict, step=epoch)
            
            # Clear local_datasets to free memory
            del local_datasets
            
            # Force garbage collection and clear GPU cache at end of each epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        if self.args.multiprocessing:
            # Terminate Processes
            terminate_processes(task_queues, processes)

        return

    def lr_update(self, epoch: int) -> None:
        self.lr = self.args.trainer.local_lr * (self.local_lr_decay) ** (epoch)
        return
    

    def save_model(self, epoch: int = -1, suffix: str = '') -> None:
        
        base_path = self.exp_path / self.args.output_model_path
        if not base_path.parent.exists():
            base_path.parent.mkdir(parents=True, exist_ok=True)

        # If saving last N epochs, save to a dedicated folder
        if hasattr(self.args, 'save_last_n_epochs') and self.args.save_last_n_epochs > 0:
            total_rounds = self.args.trainer.global_rounds
            start_save_epoch = total_rounds - self.args.save_last_n_epochs
            if epoch >= start_save_epoch:
                # Save to last_epochs folder
                last_epochs_dir = base_path.parent / "last_epochs"
                last_epochs_dir.mkdir(parents=True, exist_ok=True)
                model_path = last_epochs_dir / f"{base_path.stem}_e{epoch+1}{base_path.suffix}"
            else:
                # Regular save location
                if epoch >= 0:
                    model_path = base_path.with_name(f"{base_path.stem}_e{epoch+1}{base_path.suffix}")
                else:
                    model_path = base_path
        else:
            # Regular save with epoch suffix to avoid overwrite
            if epoch >= 0:
                model_path = base_path.with_name(f"{base_path.stem}_e{epoch+1}{base_path.suffix}")
            else:
                model_path = base_path

        if suffix:
            model_path = Path(f"{model_path}.{suffix}")
        
        save_checkpoint(self.model, model_path, epoch, save_torch=True, use_breakpoint=False)
        print(f"Saved model at {model_path}")
        return
    

    def load_model(self) -> None:
        if self.args.get('load_model_path'):
            saved_dict = torch.load(self.args.load_model_path)
            self.model.load_state_dict(saved_dict['model_state_dict'], strict=False)
            # 默认从保存的 epoch+1 继续；若显式请求 reset_start_round 再重置
            if self.args.get('reset_start_round'):
                self.start_round = 0
                logger.warning(f"Load model from {self.args.load_model_path}, reset_start_round -> 0 (saved epoch {saved_dict.get('epoch')})")
            else:
                self.start_round = saved_dict["epoch"]+1
                logger.warning(f"Load model from {self.args.load_model_path}, epoch {saved_dict['epoch']}")
            
        return


    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def validate(self, epoch: int, ) -> Dict:
        return

    def evaluate(self, epoch: int, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:

        eval_model = copy.deepcopy(self.model)
        results = self.evaler.eval(model=eval_model, epoch=epoch)
        acc = results["acc"]
        
        # Clean up evaluation model
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wandb_dict = {
            f"acc/{self.args.dataset.name}": acc,
            }

        logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')

        plt.close()
        
        self.wandb_log(wandb_dict, step=epoch)
        return {
            "acc": acc
        }
    



    
