
# Default settings on each dataset
python evaluate_nsmp.py \
    --device cuda:0 \
    --task_folder data/FB15k-237-EFO1 \
    --batch_size_eval_dataloader 1 \
    --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
    --reasoner nsmp \
    --depth_shift 1 \
    --alpha 100 \
    --llambda 0.3 \
    --prefix nsmp_lambda0.3_alpha100_ds1

python evaluate_nsmp.py \
    --device cuda:0 \
    --task_folder data/NELL-EFO1 \
    --batch_size_eval_dataloader 1 \
    --checkpoint_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
    --reasoner nsmp \
    --depth_shift 1 \
    --alpha 1000 \
    --llambda 0.1 \
    --prefix nsmp_lambda0.1_alpha1000_ds1

# Summarize the MRR results from log files
python print_mrr.py