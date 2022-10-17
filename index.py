from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import json
import h5py
import array
import pickle
import argparse
import numpy as np
from os.path import join as oj
from tqdm import tqdm, trange
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, IterableDataset

from utils import set_seed, check_dir_exist_or_build, json_dumps_arguments

from splade_models import Splade

from libs import StreamIndexDataset, CollateClass, IndexDictOfArray


def indexing(args):
    
    model = Splade(model_type_or_dir=args.pretrained_doc_encoder_path)
    model.to(args.device)
    
    indexing_batch_size = args.per_gpu_index_batch_size 
    indexing_dataset = StreamIndexDataset(args.collection_path)
    collate_func = CollateClass(args, model.transformer_rep.tokenizer)
    train_dataloader =  DataLoader(indexing_dataset, 
                                   batch_size=indexing_batch_size, 
                                   collate_fn=collate_func.collate_fn)

    dim_voc = model.transformer_rep.transformer.config.vocab_size
    sparse_index = IndexDictOfArray(args.output_index_dir_path, dim_voc=dim_voc, force_new=True)    
    count = 0
    doc_ids = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(train_dataloader, desc="Indexing", position=0, leave=True):
            inputs = {k: v.to(args.device) for k, v in batch.items() if k not in {"id"}}
            batch_documents = model(d_kwargs=inputs)["d_rep"]
            
            row, col = torch.nonzero(batch_documents, as_tuple=True)
            data = batch_documents[row, col]
            row = row + count

            batch_ids = list(batch["id"])
            doc_ids.extend(batch_ids)
            count += len(batch_ids)
            sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                               n_docs=len(batch_ids))


    sparse_index.save()
    pickle.dump(doc_ids, open(oj(args.output_index_dir_path, "doc_ids.pkl"), "wb"))
    logger.info("Done iterating over the corpus!")
    logger.info("index contains {} posting lists".format(len(sparse_index)))
    logger.info("index contains {} documents".format(len(doc_ids)))
       


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, choices=["CAsT-19", "CAsT-20", "QReCC", "TopiOCQA"])
    parser.add_argument("--collection_path", type=str, required=True, help="Path of the collection.")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, required=True, help="Path of the pretrained doc encoder.")
    
    parser.add_argument("--output_index_dir_path", type=str, required=True, help="Dir path of the output index.")
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--per_gpu_index_batch_size", type=int, required=True, help="Per gpu batch size")

    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")


    args = parser.parse_args()
    # pytorch parallel gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    check_dir_exist_or_build([args.output_index_dir_path], force_emptying=args.force_emptying_dir)
    json_dumps_arguments(oj(args.output_index_dir_path, "parameters.txt"), args)
        
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args)

    indexing(args)