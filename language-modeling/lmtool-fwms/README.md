## Code for Language Modeling Task in Our Paper

## Requirements
This toolkit requires PyTorch `torch` and Ninja `ninja` (to compile the cuda kernels).

The experiments for the paper were conducted with Python 3.6 and PyTorch >= 1.4.0.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

## Instructions

Run `sh getdata.sh` to download the data.

### Training

Softmax
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data {your data directory}/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name softmax --work_dir {your directory}
```

MGK
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data {your data directory}/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk --work_dir {your directory}
```

MLK
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data {your data directory}/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 400 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --project_name 'mgk' --seed 1111 --job_name mlk --work_dir 
```

### Evaluation

Train/valid perplexity values displayed during training are not exact in general, in the sense that the part of the text which does not fit to batch-size/backpropagation-span is discarded. In addition, for models with a limited context size, the perplexity is computed by splitting the text into segments which are treated independently (during training). The model thus has no context at the beginning of a new segment. This is avoided by using a sliding window. The commands to run the evaluation of a trained model on the test and validation set are given below.

```
CUDA_VISIBLE_DEVICES=0 python ./src/eval_sliding_window.py --cuda --data {your data directory}/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir {model_dir}
CUDA_VISIBLE_DEVICES=0 python ./src/eval_sliding_window.py --cuda --data {your data directory}/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir {model_dir}
```