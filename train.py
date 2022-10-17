from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')
sys.path.append('.')
import time
import numpy as np
import argparse
from os.path import join as oj
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import check_dir_exist_or_build, set_seed, json_dumps_arguments
from convsearch_dataset import QReCCDataset, CAsTDataset, TopiOCQADataset
from splade_models import MySeparateSplade
from libs import MixedPrecisionManager
from loss_funcs import init_regularizer, cal_kd_loss, cal_mae_loss, cal_ranking_loss


def init_simple_bert_optim(model, lr, weight_decay, warmup_steps, num_training_steps):
    """
    inspired from https://github.com/ArthurCamara/bert-axioms/blob/master/scripts/bert.py
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler


def save_model(model_output_path, model, query_tokenizer, epoch, step):
    output_dir = oj(model_output_path, 'epoch-{}'.format(epoch))
    check_dir_exist_or_build([output_dir])
    model.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))


def train_splade(args):
    if not args.need_output:
        args.log_path = "./tmp"
    if dist.get_rank() == 0:
        check_dir_exist_or_build([args.log_path], args.force_emptying_dir)
        log_writer = SummaryWriter(log_dir = args.log_path)
    else:
        log_writer = None

    # Load model
    query_model = MySeparateSplade(model_dir_path=args.pretrained_query_encoder_path, 
                                   denoising_type=args.denoising_type,
                                   num_denoising_tokens=args.num_denoising_tokens)
    doc_model = MySeparateSplade(model_dir_path=args.pretrained_doc_encoder_path)
    query_tokenizer = query_model.tokenizer
    doc_tokenizer = doc_model.tokenizer
    query_model.to(args.device)
    doc_model.to(args.device)

    if args.asr_loss_weight > 0 or args.tpd_loss_weight > 0:
        oracle_query_model = MySeparateSplade(model_dir_path=args.teacher_query_encoder_path, is_teacher=True)    # no need denoising for the teacher model
        oracle_query_model.to(args.device)
    else:
        oracle_query_model = None

    if args.n_gpu > 1:
        query_model = DDP(query_model, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)        
        dist.barrier()

    # dataloader
    Datasets = {
        "qrecc": QReCCDataset,
        "cast19": CAsTDataset,
        "cast20": CAsTDataset,
        "cast21": "NeedToImplement",
        "topiocqa": TopiOCQADataset
    }
    train_dataset = Datasets[args.dataset](args, query_tokenizer, doc_tokenizer, args.train_file_path, need_doc_info=args.need_doc_info)
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    if args.n_gpu > 1:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, 
                                batch_size = args.per_gpu_train_batch_size, 
                                sampler=sampler, 
                                collate_fn=train_dataset.get_collate_fn(args))
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    
    if args.n_gpu > 1:
        param_model = query_model.module.transformer
    else:
        param_model = query_model.transformer
    optimizer, scheduler = init_simple_bert_optim(param_model,
                                                  lr=args.learning_rate, 
                                                  warmup_steps=args.num_warmup_steps,
                                                  weight_decay=args.weight_decay,
                                                  num_training_steps=total_training_steps)

    sparsity_regularizer = init_regularizer(args.sparsity_regularization_type)

    # begin to train
    logger.info("Start training...")
    logger.info("Total training samples = {}".format(len(train_dataset)))
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))
    args.model_save_steps = max(1, int(args.model_save_steps * num_steps_per_epoch))
    args.log_print_steps = max(1, int(args.log_print_steps * num_steps_per_epoch))
    
    cur_step = 0
    use_fp16 = query_model.module.fp16 if args.n_gpu > 1 else query_model.fp16
    mpm = MixedPrecisionManager(use_fp16, "cuda" in str(args.device))
    optimizer.zero_grad()
    epoch_iterator = trange(args.num_train_epochs, desc="Epoch")
    dist.barrier()
    for epoch in epoch_iterator:
        if args.n_gpu > 1:
            query_model.module.transformer.train()
            train_loader.sampler.set_epoch(epoch)
        else:
            query_model.transformer.train()
        doc_model.transformer.eval()
        if oracle_query_model:
            oracle_query_model.eval()

        for batch in tqdm(train_loader,  desc="Step"):
            with mpm.context():
                inputs = {"input_ids": batch['bt_concat'].to(args.device), "attention_mask": batch['bt_concat_mask'].to(args.device)}
                if args.denoising_type == "oct":
                    inputs['cur_utt_end_positions'] = batch['bt_cur_utt_end_positions']
                lexical_reps, dense_reps = query_model(**inputs)  # B * dim
                
                asr_loss = torch.tensor(0.0).to(args.device)
                tpd_loss = torch.tensor(0.0).to(args.device)
                ranking_loss = torch.tensor(0.0).to(args.device)
                # training loss
                if args.asr_loss_weight > 0 or args.tpd_loss_weight > 0:
                    oracle_inputs = {"input_ids": batch["bt_oracle_utt"].to(args.device), "attention_mask": batch["bt_oracle_utt_mask"].to(args.device)}
                    # freeze oracle query encoder's parameters
                    with torch.no_grad():
                        teacher_lexical_reps, teacher_dense_reps = oracle_query_model(**oracle_inputs)
                        teacher_lexical_reps = teacher_lexical_reps.detach()
                        teacher_dense_reps = teacher_dense_reps.detach()
                    if args.asr_loss_weight > 0:
                        asr_loss = args.asr_loss_weight * cal_mae_loss(lexical_reps, teacher_lexical_reps)
                    if args.tpd_loss_weight > 0:
                        tpd_loss =  args.tpd_loss_weight * cal_kd_loss(dense_reps, teacher_dense_reps)
                if args.add_ranking_loss:
                    # doc encoder's parameters are frozen
                    with torch.no_grad():
                        doc_pos_inputs = {"input_ids": batch['bt_pos_docs'].to(args.device), "attention_mask": batch['bt_pos_docs_mask'].to(args.device)}
                        pos_doc_lexical_reps, _ = doc_model(**doc_pos_inputs) 
                        pos_doc_lexical_reps = pos_doc_lexical_reps.detach()
                        if len(batch['bt_neg_docs']) == 0:  # only_in_batch negative
                            neg_doc_lexical_reps = None
                        else:
                            batch_size, neg_ratio, seq_len = batch['bt_neg_docs'].shape       
                            batch['bt_neg_docs'] = batch['bt_neg_docs'].view(batch_size * neg_ratio, seq_len)        
                            batch['bt_neg_docs_mask'] = batch['bt_neg_docs_mask'].view(batch_size * neg_ratio, seq_len)             
                            doc_neg_inputs = {"input_ids": batch['bt_neg_docs'].to(args.device), "attention_mask": batch['bt_neg_docs_mask'].to(args.device)}
                            neg_doc_lexical_reps, _ = doc_model(**doc_neg_inputs)
                            neg_doc_lexical_reps = neg_doc_lexical_reps.detach()  # (B * neg_ratio) * dim
                    ranking_loss = cal_ranking_loss(lexical_reps, pos_doc_lexical_reps, neg_doc_lexical_reps)

                loss = asr_loss + tpd_loss + ranking_loss
                    
                # additional regularization loss
                reg_loss = torch.tensor(0.0).to(args.device)
                if sparsity_regularizer:
                    lambda_q = args.q_reg_weight
                    reg_loss = sparsity_regularizer(lexical_reps * lambda_q)
                    loss = loss + reg_loss
                    
            mpm.backward(loss)
            mpm.step(optimizer)
            scheduler.step() 

            # print info
            if dist.get_rank() == 0 and cur_step % args.log_print_steps == 0:
                logger.info("epoch = {}, \
                             current step = {}, \
                             total step = {},  \
                             total loss = {}, \
                             asr_loss = {}, \
                             tpd_loss = {}, \
                             ranking_loss = {}, \
                             reg_loss = {}".format(
                                epoch,
                                cur_step,
                                total_training_steps,
                                round(loss.item(), 5),
                                round(asr_loss.item(), 5),
                                round(tpd_loss.item(), 5),
                                round(ranking_loss.item(), 5),
                                round(reg_loss.item(), 5))
                            )
            if dist.get_rank() == 0:
                log_writer.add_scalar("total_trainig_loss", loss, cur_step)
                if args.asr_loss_weight > 0:
                    log_writer.add_scalar("asr_loss", asr_loss, cur_step)
                if args.tpd_loss_weight > 0:
                    log_writer.add_scalar("tpd_loss", tpd_loss, cur_step)
                if args.add_ranking_loss > 0:
                    log_writer.add_scalar("ranking_loss", ranking_loss, cur_step)
            cur_step += 1    # avoid saving the model of the first step.
            dist.barrier()
            # Save model
            if dist.get_rank() == 0 and args.need_output and cur_step % args.model_save_steps == 0:
                if args.n_gpu > 1:
                    save_model(args.model_output_path, query_model.module.transformer, query_tokenizer, epoch, cur_step)
                else:
                    save_model(args.model_output_path, query_model.transformer, query_tokenizer, epoch, cur_step)
                
    logger.info("Training finish!")          
    if dist.get_rank() == 0:   
        log_writer.close()
       


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser.add_argument('--n_gpu', type=int, default=1, help='The number of used GPU.')
    parser.add_argument("--dataset", type=str, required=True, choices=["cast19", "cast20", "cast21", "qrecc", "topiocqa"])
    
    parser.add_argument("--model_type", default="SPLADE", type=str)
    parser.add_argument("--denoising_type", default=None, choices=[None, "oct", "ptg"])
    parser.add_argument("--num_denoising_tokens", default=-1, type=int, help="for ptg only")
    parser.add_argument("--pretrained_query_encoder_path", type=str, required=True, help="Path of the pretrained query encoder.")
    parser.add_argument("--teacher_query_encoder_path", type=str, help="Path of the teacher ad-hoc SPLADE query encoder.")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, required=True, help="Path of the pretrained doc encoder.")
    parser.add_argument("--train_file_path", type=str, required=True, help="Path of the training dialog file.")
    parser.add_argument("--log_path", type=str, required=True, help="Path of output tensorboard log.")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path of saved models.")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Dir path of the output info.")
    parser.add_argument("--need_output", action="store_true", help="Whether need to output logs and models (creating the dirs)")
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    parser.add_argument("--log_print_steps", type=float, default=0.01, help="Percent of steps per epoch to print once.")
    parser.add_argument("--model_save_steps", type=float, required=True, help="Percent of steps to save the model once")

    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Training epochs")
    parser.add_argument("--per_gpu_train_batch_size", type=int, required=True, help="Per gpu batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    # parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Warm up steps.")

    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, required=True, help="Max response length, 100 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, required=True, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--enable_last_response", action="store_true", help="True for CAsT-20")

    parser.add_argument("--asr_loss_weight", type=float, default=0.0, help="adaptive sparsity regularization weight")
    parser.add_argument("--tpd_loss_weight", type=float, default=0.0, help="teacher proxy distilltion loss weight")
    parser.add_argument("--add_ranking_loss", action="store_true", help="add (lexical) ranking loss")
    parser.add_argument("--collate_fn_type", type=str, required=True, choices=["flat_concat_for_train"], help="To control how to organize the batch data.")
    parser.add_argument("--negative_type", type=str, required=True, choices=["random_neg", "bm25_hard_neg", "prepos_hard_neg", "in_batch_neg"])
    parser.add_argument("--neg_ratio", type=int, help="negative ratio")
    parser.add_argument("--need_doc_info", action="store_true", help="Whether need doc info or not.")

    parser.add_argument("--sparsity_regularization_type", type=str, required=True, choices=["None", "L1", "FLOPS"])
    parser.add_argument("--q_reg_weight", type=float, required=True, help="sparsity weight for query encoder.")


    args = parser.parse_args()
    local_rank = args.local_rank
    args.local_rank = local_rank
    # pytorch parallel gpu  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
    args.device = device
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    if dist.get_rank() == 0 and args.need_output:
        check_dir_exist_or_build([args.output_dir_path], force_emptying=args.force_emptying_dir)
        json_dumps_arguments(oj(args.output_dir_path, "parameters.txt"), args)
        

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    train_splade(args)
