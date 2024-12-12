export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="./datasets/imagenet/"
MODEL_PTH="./pretrained"

# eval
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --epoch 30 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --distill True --eval --if_merge_odd
# retrain
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29449 main.py --lr 2e-5 --min-lr 1e-6 --epoch 30 --data-path $DATA_PATH --output_dir ./log --model RMeeTo_tiny --merge_method ToMe --num_prune 5 --metric X --if_order --model_pth $MODEL_PTH --batch-size 128 --distance cosine --distill True --eval --if_merge_odd