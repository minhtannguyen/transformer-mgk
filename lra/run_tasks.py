from fvcore.nn import FlopCountAnalysis
from model_wrapper import ModelForSC, ModelForSCDual, ModelForSCProbing, ModelForSCDualProbing
from dataset import LRADataset
import torch
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import itertools
 
import wandb

 
# assert 1==2
 
# 1. Start a new run
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model", dest="model", required=True)
parser.add_argument("--task", type=str, help="task", dest="task", required = False)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument("--logging", action='store_true', default=False)
parser.add_argument("--expname", type=str, default="default")
parser.add_argument("--multi_gauss", type=bool, default=False)
parser.add_argument("--save_attn", type=bool, default=False)
parser.add_argument("--split_attn_type", type = bool, default= False)
parser.add_argument("--fixed_pos_emb", type= bool, default= False)
parser.add_argument("--norm_qk", type= bool, default= False)
parser.add_argument("--mg_layer1", type= bool, default= False)
parser.add_argument("--mg_layer2", type= bool, default= False)
parser.add_argument("--accu_grad_step", type= int, default= 1)
parser.add_argument("--key2", type= bool, default= False)
parser.add_argument("--hard_em", type= bool, default= False)
parser.add_argument("--soft_em", type= bool, default= False)
parser.add_argument("--l2_reg", type= bool, default= False)
parser.add_argument("--track_kk", type= bool, default= False)
parser.add_argument("--add_performer", type= bool, default= False)
 
# Model configs
parser.add_argument("--attention_grad_checkpointing", default=False, action="store_true")
parser.add_argument("--num_landmarks", default=128, type=int)
parser.add_argument("--window_size", default=129, type=int)
parser.add_argument("--conv_kernel_size", default=-1, type=int)
parser.add_argument("--learn_pos_emb", default=1, type=int,
                    help="Use 0 or 1 to represent false and true")
parser.add_argument("--tied_weights", default=False, action="store_true")
parser.add_argument("--embedding_dim", default=64, type=int)
parser.add_argument("--transformer_dim", default=64, type=int)
parser.add_argument("--transformer_hidden_dim", default=128, type=int)
parser.add_argument("--head_dim", default=32, type=int)
parser.add_argument("--num_head", default=2, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--vocab_size", default=512, type=int)
parser.add_argument("--max_seq_len", default=4096, type=int)
parser.add_argument("--dropout_prob", default=0.1, type=float)
parser.add_argument("--attention_dropout", default=0.1, type=float)
parser.add_argument("--pooling_mode", default="MEAN", type=str)
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--cls_token", default=False, action='store_true')
 
 
# Training configs
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--warmup", default=8000, type=int)
parser.add_argument("--lr_decay", default="linear", type=str)
parser.add_argument("--fixed_lr", default=False, action='store_true')
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--adam_eps", default=1e-6, type=float)
 
parser.add_argument("--eval_frequency", default=500, type=int)
parser.add_argument("--num_train_steps", default=20000, type=int)
parser.add_argument("--num_eval_steps", default=781, type=int)
parser.add_argument("--fp32_attn", default=False, action='store_true')
parser.add_argument("--conv_zero_init", default=False, action='store_true')
 
# Dataset Configs
parser.add_argument("--n_train_samples", default=25000, type=int)
parser.add_argument("--n_dev_samples", default=25000, type=int)
parser.add_argument("--n_test_samples", default=25000, type=int)
 
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--cls_last_layer", default=False, action='store_true')
 
parser.add_argument("--seed", default=1234, type=int)
 
parser.add_argument("--linformer_k", default=256, type=int)
parser.add_argument("--rp_dim", default=256, type=int)
parser.add_argument("--num_hash", default=2, type=int)
parser.add_argument("--chk_path", default="LRA_chks", type=str)
parser.add_argument("--test_flops", default=False, action='store_true')
parser.add_argument('--pi0', type= float, default= 0.9)
args = parser.parse_args()
 
from datetime import datetime
 
now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
 
multigauss = args.multi_gauss
norm_qk = args.norm_qk
 
model_setup = f"{current_time}_{args.model}_{args.task}_{args.seed}_head{args.num_head}_layers{args.num_layers}"
if args.model =='linear' and args.add_performer:
    model_setup = model_setup + '_add_performer'
if args.key2:
    model_setup =  model_setup + f"_key2:{args.pi0}"
if norm_qk:
    model_setup = model_setup + "_norm_qk"
if args.fixed_pos_emb:
    model_setup = model_setup + "_fixed_pe"
if args.mg_layer1 and multigauss:
    model_setup = model_setup + f"_mg_layer1:{args.pi0}"
elif args.mg_layer2 and multigauss:
    model_setup = model_setup + f"_mg_layer2:{args.pi0}"
elif multigauss:
    model_setup = model_setup + f"_multigauss:{args.pi0}"
if args.split_attn_type:
    model_setup = model_setup + '_split'
if args.key2 or args.multi_gauss:
    if args.hard_em:
        model_setup = model_setup + '_hard_em'
    elif args.soft_em:
        model_setup = model_setup + '_soft_em'
    else:
        model_setup = model_setup + '_GDpi'
        if args.track_kk:
            model_setup = model_setup + '_track_kk'
    if args.l2_reg:
        model_setup = model_setup + '_l2'
 


 
def seed_all(seed_value):
    torch.manual_seed(seed_value)
     # cpu  vars
    np.random.seed(seed_value)
    random.seed(seed_value)
 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
seed_all(args.seed)
# random.seed(args.seed)
# torch.manual_seed(args.seed)
# cudnn.deterministic = True
 
args.attn_type = args.model # remove attn_type in the future
args.mixed_precision = True # bool(args.mixed_precision)
task = args.task
 
checkpoint_dir = args.chk_path
print(args)
 
device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")
 
if task == "retrieval":
    if args.test_flops:
        model = ModelForSCDualProbing(args)
    else:
        model = ModelForSCDual(args)
else:
    if args.test_flops:
        model = ModelForSCProbing(args)
    else:
        model = ModelForSC(args)
 
# print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush=True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush=True)
 
model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)
print(model)
# assert 1==2
data_path = '../../../Nystromformer/LRA/datasets'
 
##im suing batch_size of test and dev difers from train for retrieval task
ds_iter = {
    "train":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.train.pickle", True), batch_size=args.batch_size, drop_last=True)),
    "dev":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.dev.pickle", True), batch_size=args.batch_size, drop_last=True)),
    "test":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", False), batch_size=args.batch_size, drop_last=True)),
}
 
 
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999), eps=args.adam_eps, weight_decay=args.weight_decay
)
 
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=args.learning_rate,
    pct_start=args.warmup / args.num_train_steps,
    anneal_strategy=args.lr_decay,
    total_steps=args.num_train_steps
)
 
amp_scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
 
def step(component, step_idx):
    t0 = time.time()
 
    # optimizer.zero_grad()
 
    _, batch = next(ds_iter[component])
 
    for key in batch:
        batch[key] = batch[key].cuda()
 
    if (args.model == 'nystrom' or args.model == 'reformer') and args.pooling_mode.lower() == 'cls':
        for key in batch:
            if 'input_ids' in key or 'mask' in key:
                batch[key] = batch[key][:, :-1].contiguous()
 
    if component == "train":
        outputs = {}
 
        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp
 
        for partial_inputs in partial_inputs_list:
            if args.test_flops:
                if 'input_ids_1' in partial_inputs:
                    flops = FlopCountAnalysis(
                        model, [partial_inputs['input_ids_0'][:1], partial_inputs['input_ids_1'][:1],
                                partial_inputs['mask_0'][:1], partial_inputs['mask_1'][:1], partial_inputs['label'][:1]])
                else:
                    flops = FlopCountAnalysis(
                        model, [partial_inputs['input_ids_0'][:1], partial_inputs['mask_0'][:1], partial_inputs['label'][:1]])
 
                print(f"Flops of {args.model}: {flops.total()/1e9:.2f} G")
                exit()
 
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()
        # if (step_idx + 1) % args.accu_grad_step == 0:
        amp_scaler.step(optimizer)
        amp_scaler.update()
        optimizer.zero_grad()
 
        if (not args.fixed_lr) or step_idx < args.warmup:
            lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}
 
            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp
 
            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]
 
    t1 = time.time()
 
    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t
 
    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)
 
    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)
 
    if args.save_attn:
        return batch
 
  
 
def print_summary(summary, save_if_improved, train_step_idx, subset):
    # subset: str, the subset to report the result
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])
 
    _loss = np.mean(summary["loss"])
    _acc = np.mean(summary["accu"])
    
 
    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
            print(f"best_accu={best_accu}. Saved best model")
 
    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key+f"_{subset}"] = summary[key]
        else:
            summary_round[key+f"_{subset}"] = round(summary[key], 4)
 
    print(summary_round, flush=True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()
 
 
 
    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []
 
    return _loss, _acc
# model_task_seed_head_dim__num_layers
init_t = time.time()
 
log_f_path = os.path.join(checkpoint_dir, model_setup + "_output.log")
log_f = open(log_f_path, "a+")
print(model_setup)
 
 
summary = {
    component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
    for component in ["train", "dev", "test"]
}
 
# accumu_steps = max(training_config["batch_size"] // len(device_ids) // gpu_memory_config[attn_type], 1)
accumu_steps = max(args.batch_size // len(device_ids) // 32, 1)
print(f"accumu_steps={accumu_steps}")
 
from collections import defaultdict
 
# temp = os.path.join(checkpoint_dir, '21_09_2021_08:44:52_softmax_retrieval_4096_head4_layers2_key2:0.0_GDpi' + "_output.model")
# checkpoint = torch.load(temp)
# model.module.load_state_dict(checkpoint["model_state_dict"])
# model.cuda()
 
save_metrics = defaultdict(list)
if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(args.num_train_steps):
            outputs = step("train", train_step_idx)
 
            if (train_step_idx + 1) % args.eval_frequency == 0:
                train_loss, train_acc = print_summary(summary["train"], False, train_step_idx, 'train')
                save_metrics['train_loss'].append(train_loss)
                save_metrics['train_acc'].append(train_acc)
                model.eval()

 
                for dev_step_idx in range(args.num_eval_steps):
                    outputs = step("dev", dev_step_idx)

 
                val_loss, val_acc = print_summary(summary["dev"], True, train_step_idx, 'dev')
                save_metrics['val_loss'].append(val_loss)
                save_metrics['val_acc'].append(val_acc)
                model.train()
    except KeyboardInterrupt as e:
        print(e)
 

checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location="cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
# print(model.module.model.transformer_0.mha.attn.pi.clone().detach())
# print(model.module.model.transformer_1.mha.attn.pi.clone().detach())
# assert 1==2
 
 

try:
    for test_step_idx in itertools.count():
        step("test", test_step_idx)
        # if args.save_attn:
        #     attn_matrices = [model.module.model.transformer_0.mha.attn.attn_matrix[:10].detach().cpu().numpy(), model.module.model.transformer_1.mha.attn.attn_matrix[:10].detach().cpu().numpy()]
except StopIteration:
    test_loss, test_acc = print_summary(summary["test"], False, 0, 'test')
    save_metrics['test_loss'].append(test_loss)
    save_metrics['test_acc'].append(test_acc)
 

 
# if args.save_attn:
#     for test_step_idx in itertools.count():
#         batch = step("test", test_step_idx)
#         attn_matrices = [model.module.model.transformer_0.mha.attn.attn_matrix[:10].detach().cpu().numpy(), model.module.model.transformer_1.mha.attn.attn_matrix[:10].detach().cpu().numpy()]  
#         break
# # if args.save_attn:
#     import pickle
#     with open(f'attention_{args.task}.pkl', 'rb') as f:
#         data = pickle.load(f)
#     # data = {model_setup:{'attn_matrices':attn_matrices, 'batch': batch['input_ids_0'][:10]}}
#     data.update({model_setup:{'attn_matrices':attn_matrices, 'batch': batch['input_ids_0'][:10]}})
#     print(f'attention_{args.task}.pkl')
#     with open(f'attention_{args.task}.pkl', 'wb') as f:
#         pickle.dump(data, f)

 
 

