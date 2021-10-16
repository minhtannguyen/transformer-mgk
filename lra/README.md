Softmax 1 heads
```
CUDA_VISIBLE_DEVICES=0 python run_tasks.py --model softmax --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 
```

#key2 1 head
```
CUDA_VISIBLE_DEVICES=0 python run_tasks.py --model softmax --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 --key2 True 
```

#rbf2 1 head
```
CUDA_VISIBLE_DEVICES=1 python run_tasks.py --model softmax --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 --multi_gauss True 
```


#Hard_E for 2keys
```
CUDA_VISIBLE_DEVICES=0 python run_tasks.py --model softmax --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 --key2 True --hard_em True
```

#Soft_E for 2keys
```
CUDA_VISIBLE_DEVICES=0 python run_tasks.py --model softmax --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 --key2 True --soft_em True
```


##Linear 1 heads
```
CUDA_VISIBLE_DEVICES=0 python run_tasks.py --model linear --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 
```

#key2 Linear 1 head
```
CUDA_VISIBLE_DEVICES=0 python run_tasks.py --model linear --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 --key2 True 
```

#rbf2 Linear 1 head
```
CUDA_VISIBLE_DEVICES=1 python run_tasks.py --model linear --task retrieval --pi0 0.0 --batch_size 24 --warmup 8000 --learning_rate 0.0001 --weight_decay 0.0 --num_layers 2  --n_train_samples 147086 --n_dev_samples 18090 --n_test_samples 17437 --max_seq_len 4096 --num_train_steps 30000 --num_eval_steps 565 --eval_frequency 300 --seed 4096 --num_head 1 --multi_gauss True 
```

