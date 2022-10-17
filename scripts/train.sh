model_name="lecore"
dataset="qrecc"
output_dir_path="./experiments/train/"$dataset/$model_name
n_gpu=2
negative_type="random_neg"
neg_ratio=1
epochs=4

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port 23452 \
train.py \
--n_gpu=$n_gpu \
--dataset=$dataset \
--asr_loss_weight=1e-4 \
--tpd_loss_weight=0.1 \
--denoising_type="ptg" \
--num_denoising_tokens=48 \
--add_ranking_loss \
--pretrained_query_encoder_path="naver/efficient-splade-V-large-query" \
--teacher_query_encoder_path="naver/efficient-splade-V-large-query" \
--pretrained_doc_encoder_path="naver/efficient-splade-V-large-doc" \
--train_file_path="./datasets/$dataset/preprocessed/train_with_$negative_type.json" \
--output_dir_path=$output_dir_path \
--log_path=$output_dir_path"/log" \
--model_output_path=$output_dir_path"/checkpoints" \
--log_print_steps=0.1 \
--model_save_steps=1.0 \
--use_data_percent=1.0 \
--num_train_epochs=$epochs \
--per_gpu_train_batch_size=64 \
--max_doc_length=256 \
--max_concat_length=256 \
--max_response_length=64 \
--collate_fn_type="flat_concat_for_train" \
--negative_type=$negative_type \
--neg_ratio=$neg_ratio \
--sparsity_regularization_type="None" \
--q_reg_weight=0.0 \
--force_emptying_dir \
--need_doc_info \
--need_output \



