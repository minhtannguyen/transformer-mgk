#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name softmax-seed-1111 --work_dir /tanData/mgattn/softmax-seed-1111

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-soft-e-n-head-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-soft-e-n-head-seed-1111


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-hard-e-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-hard-e-seed-1111


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-soft-e-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-soft-e-seed-1111


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-rbf-pi-reg-0-nhead-klen-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-rbf-pi-reg-0-nhead-klen-seed-1111 --pi_reg 0.0

CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-soft-e-nhead-klen-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-soft-e-nhead-klen-seed-1111

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-rbf-pi-reg-0-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-rbf-pi-reg-0-nhead-klen-4head-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-seed-1111 --pi_reg 0.0




CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small4x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small4x-seed-1111 --pi_reg 0.0

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small8x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small8x-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small1-5x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small1-5x-seed-1111 --pi_reg 0.0





CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-soft-pi-reg-0-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-soft-pi-reg-0-nhead-klen-4head-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-uniinit-pi-reg-0-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-uniinit-pi-reg-0-nhead-klen-4head-seed-1111 --pi_reg 0.0 --md_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-rbf-pi-reg-0-md-reg-0-01-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-rbf-pi-reg-0-md-reg-0-01-nhead-klen-4head-seed-1111 --pi_reg 0.0 --md_reg 0.01



CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name debug --work_dir /tanData/mgattn/debug --pi_reg 0.0 --md_reg 0.01




CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name debug1 --work_dir /tanData/mgattn/debug1 --log-interval 1

