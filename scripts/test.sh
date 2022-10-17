experiment_tag="lecore_test"
experiment_dir=$experiment_tag
index_dir_path="./indexes/qrecc/original_splade/"
test_input_type="flat_concat"
dataset="qrecc"
model_name="lecore"
epoch="epoch-"$1
checkpoint_path="./experiments/train/qrecc/$model_name/checkpoints/$epoch"

export CUDA_VISIBLE_DEVICES=2
python test.py --dataset=$dataset \
--test_input_type=$test_input_type \
--collate_fn_type="flat_concat_for_test" \
--test_file_path="./datasets/qrecc/preprocessed/test.json" \
--gold_qrel_file_path="./datasets/qrecc/preprocessed/qrecc_qrel.tsv" \
--query_encoder_checkpoint=$checkpoint_path \
--index_dir_path=$index_dir_path \
--eval_batch_size=32 \
--max_concat_length=256 \
--max_response_length=64 \
--rel_threshold=1 \
--retrieval_output_path="./experiments/test/$dataset/$model_name/$epoch" \
--use_data_percent=1.0 \
--force_emptying_dir \



