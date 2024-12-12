export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="./k400/data_list"
PREFIX="./k400/OpenMMLab___Kinetics-400/raw/Kinetics-400"

python -m torch.distributed.run --nproc_per_node=4 \
    main.py \
    --model videomamba_R_MeeTo_tiny \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'Kinetics_sparse' \
    --split ',' \
    --nb_classes 400 \
    --log_dir ./log \
    --output_dir ./output \
    --batch_size 32 \
    --update_freq 2 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --num_workers 12 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 30 \
    --lr 2e-5 \
    --drop_path 0.1 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --num_prune 40 \
    --if_order \
    --bf16