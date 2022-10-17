from IPython import embed
import json
import os
import sys
sys.path.append('..')
sys.path.append('.')
from tqdm import tqdm
import random
import logging
from utils import pstore, pload
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def gen_qrecc_passage_collection(input_passage_dir, output_file, pid2rawpid_path):
    '''
    - input_passage_dir = "collection-paragraph"
    - output_file = "qrecc_collection.tsv"
    - pid2rawpid_path = "pid2rawpid.pkl"
    '''
    def process_qrecc_per_dir(dir_path, pid, pid2rawpid, fw):
        filenames = os.listdir(dir_path)
        for filename in tqdm(filenames):
            with open(os.path.join(dir_path, filename), "r") as f:
                data = f.readlines()
            for line in tqdm(data):
                line = json.loads(line)
                raw_pid = line["id"]
                passage = line["contents"]
                pid2rawpid[pid] = raw_pid
                fw.write("{}\t{}".format(pid, passage))
                fw.write("\n")
                
                pid += 1
        
        return pid, pid2rawpid


    pdir1 = os.path.join(input_passage_dir, "commoncrawl")
    pdir2 = os.path.join(input_passage_dir, "wayback")
    pdir3 = os.path.join(input_passage_dir, "wayback-backfill")
    
    pid = 0
    pid2rawpid = {}

    with open(output_file, "w") as fw:
        pid, pid2rawpid = process_qrecc_per_dir(pdir1, pid, pid2rawpid, fw)
        logger.info("{} process ok!".format(pdir1))
        pid, pid2rawpid = process_qrecc_per_dir(pdir2, pid, pid2rawpid, fw)
        logger.info("{} process ok!".format(pdir2))
        pid, pid2rawpid = process_qrecc_per_dir(pdir3, pid, pid2rawpid, fw)
        logger.info("{} process ok!".format(pdir3))

    pstore(pid2rawpid, pid2rawpid_path, True)

    logger.info("generate QReCC passage collection -> {} ok!".format(output_file))
    logger.info("#totoal passages = {}".format(pid))


def gen_qrecc_qrel(input_test_file, output_qrel_file, pid2rawpid_path):
    '''
    - input_test_file = "scai-qrecc21-test-turns.json"
    - pid2rawpid_path = "pid2rawpid.pkl"
    - output_qrel_file = "qrecc_qrel.tsv"
    '''
    with open(input_test_file, "r") as f:
        data = json.load(f)

    pid2rawpid = pload(pid2rawpid_path)
    rawpid2pid = {}
    for pid, rawpid in enumerate(pid2rawpid):
        rawpid2pid[rawpid] = pid

    with open(output_qrel_file, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("QReCC-Test", line['Conversation_no'], line['Turn_no'])
            for rawpid in line['Truth_passages']:
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, rawpid2pid[rawpid], 1))
                f.write('\n') 
    
    logger.info("generate qrecc qrel file -> {} ok!".format(output_qrel_file))


def gen_qrecc_train_test_files(train_inputfile,
                               test_inputfile, 
                               train_outputfile, 
                               test_outputfile, 
                               pid2rawpid_path,
                               max_random_neg_raito = 5):
    '''
    - train_inputfile = "scai-qrecc21-training-turns.json"
    - test_inputfile = "scai-qrecc21-test-turns.json"
    - train_outputfile = "train.json"
    - test_outputfile = "test.json"
    - pid2rawpid_path = "pid2rawpid.pkl"
    '''
    pid2rawpid = pload(pid2rawpid_path)
    rawpid2pid = {}
    for pid, rawpid in enumerate(pid2rawpid):
        rawpid2pid[rawpid] = pid
    
    sid2utt = {}
    sid2pospid = {}
    
    # train & test raw files
    num_num_doc = 54573064
    outputfile2inputfile = {train_outputfile : train_inputfile,
                            test_outputfile: test_inputfile}
    for outputfile in outputfile2inputfile:
        with open(outputfile2inputfile[outputfile], "r") as f:
            data = json.load(f)
        with open(outputfile, "w") as f:
            for line in tqdm(data):
                record = {}
                sample_title = "QReCC-Train" if outputfile == train_outputfile else "QReCC-Test"
                sample_id = "{}_{}_{}".format(sample_title, line['Conversation_no'], line['Turn_no'])
                record["sample_id"] = sample_id
                record["source"] = line["Conversation_source"]
           
                cur_utt_text = line["Question"] if int(line['Turn_no']) != 1 else line["Truth_rewrite"] # according to the paper of CONQRR
                sid2utt[sample_id] = cur_utt_text
                record["cur_utt_text"] = cur_utt_text

                oracle_utt_text = line["Truth_rewrite"]
                record["oracle_utt_text"] = oracle_utt_text
                
                cur_response_text = line["Truth_answer"]
                record["cur_response_text"] = cur_response_text
                
                ctx_utts_text = []
                for i in range(0, len(line['Context'])):
                    if i % 2 == 0:
                        ctx_query_utt = sid2utt["{}_{}_{}".format(sample_title, line['Conversation_no'], int(i / 2) + 1)]                    
                        ctx_utts_text.append(ctx_query_utt)
                    else:
                        ctx_response_utt = line['Context'][i]
                        ctx_utts_text.append(ctx_response_utt)
                record["ctx_utts_text"] = ctx_utts_text
                    

                # Actually useful for training file only
                # process pos doc info, only store pos docs ids and random negative doc ids.
                # Then we will add neg doc ids and then extract doc content.
                pos_docs_pids = []        
                for rawpid in line['Truth_passages']:
                    pos_pid = rawpid2pid[rawpid]
                    pos_docs_pids.append(pos_pid)
                sid2pospid[sample_id] = pos_docs_pids
                record["pos_docs_pids"] = pos_docs_pids
                
                # Various negatives
                if outputfile == train_outputfile:
                    # 1. random negatives
                    random_neg_docs_pids = set()
                    while len(random_neg_docs_pids) < max_random_neg_raito:
                        neg_pid = random.randint(0, num_num_doc - 1)
                        if neg_pid not in pos_docs_pids:
                            random_neg_docs_pids.add(neg_pid)
                    record["random_neg_docs_pids"] = list(random_neg_docs_pids)
                    
                    # 2. previous turns positive docs as the negatives of the current turn.
                    prepos_neg_docs_pids = set()
                    for turn_id in range(1, int(line['Turn_no'])):
                        tmp_sample_id = "{}_{}_{}".format(sample_title, line['Conversation_no'], turn_id)
                        prepos_pids = sid2pospid[tmp_sample_id]
                        prepos_neg_docs_pids = prepos_neg_docs_pids | set(prepos_pids)
                    prepos_neg_docs_pids = prepos_neg_docs_pids - set(pos_docs_pids)
                    record["prepos_neg_docs_pids"] = list(prepos_neg_docs_pids)                      
                    
                f.write(json.dumps(record))
                f.write('\n')
    
    logger.info("QReCC train test file preprocessing (first stage) ok!")


def extract_doc_content_of_random_negs_for_train_file(qrecc_collection_path, 
                                       train_inputfile, 
                                       train_outputfile_with_doc,
                                       random_neg_ratio = 1):
    '''
    - qrecc_collection_path = "qrecc_collection.tsv"
    - train_inputfile = "train.json"
    - train_outputfile_with_doc = "train_with_doc.json"       
    '''
    with open(train_inputfile, 'r') as f:
        data = f.readlines()
    
    needed_pids_set = set()
    for line in tqdm(data):
        line = json.loads(line)
        needed_pids_set = needed_pids_set | set(line["pos_docs_pids"])
        needed_pids_set = needed_pids_set | set(line["random_neg_docs_pids"][:random_neg_ratio])

    # load collection.tsv
    pid2doc = {}
    num_num_doc = 54573064
    bad_doc_set = set()
    logger.info("Loading QReCC collection, total 54M passages...")
    for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
        try:
            pid, doc = line.strip().split('\t')
            pid = int(pid)
        except:
            pid = int(line.strip().split('\t')[0])
            doc = ""
            bad_doc_set.add(pid)
        if pid in needed_pids_set:
            pid2doc[pid] = doc
    logger.info("Loadding QReCC collection OK! Total bad passages = {}".format(len(bad_doc_set)))
    
    # Merge doc content to the train file
    with open(train_outputfile_with_doc, 'w') as fw:
        for line in data:
            line = json.loads(line)
            pos_docs_text = []
            for pid in line["pos_docs_pids"]:
                if pid in pid2doc:
                    pos_docs_text.append(pid2doc[pid])
            
            pos_docs_text = modify_pos_docs(line, pos_docs_text)
            line["pos_docs_text"] = pos_docs_text

            if len(pos_docs_text) > 0:
                random_neg_docs_text = []
                for pid in line["random_neg_docs_pids"][:random_neg_ratio]:
                    if pid in pid2doc:
                        random_neg_docs_text.append(pid2doc[pid])
                
                random_neg_docs_text = modify_neg_docs(line, random_neg_docs_text)
                line["random_neg_docs_text"] = random_neg_docs_text
                
            fw.write(json.dumps(line))
            fw.write('\n')

    logger.info("QReCC train file with doc (pos+neg) content are generated OK!")
          

# def modify_pos_docs(conv_sample, pos_docs_text):
#     '''
#     Modify the pos doc content based on the current conversational sample 
#     to avoid simply string-match, enhance model generalization ability
#     '''
#     return pos_docs_text

# def modify_neg_docs(conv_sample, neg_docs_text):
#     '''
#     Modify the neg doc content based on the current conversational sample 
#     to avoid simply string-match, enhance model generalization ability
#     '''
#     return neg_docs_text


if __name__ == "__main__":
    train_inputfile = "scai-qrecc21-training-turns.json"
    test_inputfile = "scai-qrecc21-test-turns.json"
    train_outputfile = "train.json"
    test_outputfile = "test.json"
    pid2rawpid_path = ""    # this is a dict whose key is reindexed-pid (passage id) and the value is the raw pid.
    gen_qrecc_train_test_files(train_inputfile, test_inputfile, train_outputfile, test_outputfile, pid2rawpid_path)
    
    qrecc_collection_path = ""    # please download this large qrecc collection from the official website
    train_inputfile = "train.json"
    train_outputfile_with_doc = "train_with_random_neg.json"
    extract_doc_content_of_random_negs_for_train_file(qrecc_collection_path, train_inputfile, train_outputfile_with_doc)

    input_test_file = "scai-qrecc21-test-turns.json"
    pid2rawpid_path = "pid2rawpid.pkl"
    output_qrel_file = "qrecc_qrel.tsv"
    gen_qrecc_qrel(input_test_file, output_qrel_file, pid2rawpid_path)
