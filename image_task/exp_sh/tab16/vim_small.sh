export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="./datasets/imagenet/"
MODEL_PTH="./pretrained"

python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29479 main.py --lr 2e-5 --min-lr 1e-6 --epoch 3 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_small --merge_method ToMe --num_prune 11 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --distill True --merge_interval 4
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29479 main.py --lr 2e-5 --min-lr 1e-6 --epoch 3 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_small --merge_method ToMe --num_prune 11 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --distill True --merge_interval 6
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29479 main.py --lr 2e-5 --min-lr 1e-6 --epoch 3 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_small --merge_method ToMe --num_prune 11 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --distill True --merge_interval 8
