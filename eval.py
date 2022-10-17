from IPython import embed
import sys
sys.path += ['..']
import json
from evaluate import eval_run_with_qrel

run_file = "/home/kelong_mao/ConvRetriever/experiments/splade/topiocqa/original_splade/oracle/run.json"
gold_qrel_file_path = "/data1/kelong_mao/datasets/topiocqa/preprocessed/topiocqa_qrel.tsv"

with open(run_file, "r") as f:
    result = json.load(f)

eval_kwargs = {"run":result, 
                "qrel_file": gold_qrel_file_path, 
                "rel_threshold": 1,
                "retrieval_output_path": "."}
eval_run_with_qrel(**eval_kwargs)

