from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import set_seed, check_dir_exist_or_build, json_dumps_arguments, get_has_gold_label_test_qid_set
from evaluate import eval_run_with_qrel

from splade_models import MySeparateSplade
from libs import SparseRetrieval
from convsearch_dataset import QReCCDataset, CAsTDataset, TopiOCQADataset



def sparse_retrieve_and_evaluate(args):
    model = MySeparateSplade(model_dir_path=args.query_encoder_checkpoint, 
                             denoising_type=args.denoising_type,
                             num_denoising_tokens=args.num_denoising_tokens)
    model.to(args.device)
    
    doc_tokenizer = None
    query_tokenizer = model.tokenizer
    
    Datasets = {
        "qrecc": QReCCDataset,
        "cast19": CAsTDataset,
        "cast20": CAsTDataset,
        "topiocqa": TopiOCQADataset
    }
    test_dataset = Datasets[args.dataset](args, query_tokenizer, doc_tokenizer, args.test_file_path, need_doc_info=False)
    test_loader = DataLoader(test_dataset, 
                            batch_size = args.eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args))
    
    # get query embeddings
    qid2emb = {}
    has_gold_label_qids = get_has_gold_label_test_qid_set(args.gold_qrel_file_path)
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, desc="Inferencing"):
            inputs = {}
            if args.test_input_type == "flat_concat":
                inputs["input_ids"] = batch["bt_concat"].to(args.device)
                inputs["attention_mask"] = batch["bt_concat_mask"].to(args.device)
                if args.denoising_type == "oct":
                    inputs["cur_utt_end_positions"] = batch["bt_cur_utt_end_positions"].to(args.device)
            elif args.test_input_type == "raw":
                inputs["input_ids"] = batch["bt_cur_utt"].to(args.device)
                inputs["attention_mask"] = batch["bt_cur_utt_mask"].to(args.device)
            elif args.test_input_type == "oracle":
                inputs["input_ids"] = batch["bt_oracle_utt"].to(args.device)
                inputs["attention_mask"] = batch["bt_oracle_utt_mask"].to(args.device)
            
            batch_query_embs, _ = model(**inputs)
            qids = batch["bt_sample_id"] 
            for i, qid in enumerate(qids):
                if qid not in has_gold_label_qids:
                    continue
                qid2emb[qid] = batch_query_embs[i]
    
    # retrieve
    dim_voc = model.transformer.config.vocab_size
    retriever = SparseRetrieval(args.index_dir_path, args.retrieval_output_path, dim_voc, args.top_n)
    result = retriever.retrieve(qid2emb)
    
    # evaluate
    eval_kwargs = {"run":result, 
                   "qrel_file": args.gold_qrel_file_path, 
                   "rel_threshold": args.rel_threshold,
                   "retrieval_output_path": args.retrieval_output_path}
    eval_run_with_qrel(**eval_kwargs)

    logger.info("Evaluation OK!")
       


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, choices=["cast19", "cast20", "qrecc", "topiocqa"])
    parser.add_argument("--model_type", default="SPLADE", type=str)
    parser.add_argument("--denoising_type", default=None, choices=[None, "oct", "ptg"])
    parser.add_argument("--num_denoising_tokens", default=-1, type=int, help="for ptg only")
    parser.add_argument("--test_input_type", type=str, required=True, choices=["flat_concat", "raw", "oracle"])
    parser.add_argument("--query_encoder_checkpoint", type=str, required = True, help="The tested conversational query encoder path.")
    parser.add_argument("--collate_fn_type", type=str, required=True, choices=["flat_concat_for_test"], help="To control how to organize the batch data. Same as in the train_model.py")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, required=True, help="Max response length, 64 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, required=True, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--enable_last_response", action="store_true", help="True for CAsT-20")

    # test input file
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--index_dir_path", type=str, required=True)
    parser.add_argument("--gold_qrel_file_path", type=str, required=True)
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")
    
    # test parameters 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=1000)

    # output file
    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    check_dir_exist_or_build([args.retrieval_output_path], force_emptying=args.force_emptying_dir)
    json_dumps_arguments(os.path.join(args.retrieval_output_path, "parameters.txt"), args)
        
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args)
    sparse_retrieve_and_evaluate(args)