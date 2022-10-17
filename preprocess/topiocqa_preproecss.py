from json.tool import main
from IPython import embed
import json
from tqdm import tqdm

def gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path):
    '''
    raw_dev_file_path = "gold_dev.json"
    output_qrel_file_path = "topiocqa_qrel.tsv"
    '''
    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    with open(output_qrel_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            for pos in line["positive_ctxs"]:
                pid = int(pos["passage_id"]) - 1
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, pid, 1))
                f.write('\n')




def gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, ouput_test_file_path):
    '''
    raw_train_file_path = "gold_train.json"
    raw_dev_file_path = "gold_dev.json"
    output_train_file_path = "train.json"
    ouput_test_file_path = "test.json"
    '''
    with open(raw_train_file_path, "r") as f:
        data = json.load(f)
    

    last_conv_id = -1
    context_queries_and_answers = []
    context_pos_docs_pids = set()

    with open(output_train_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Train", line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                pos_docs.append(pos["title"] + ". " + pos["text"])
                pos_docs_pids.append(int(pos["passage_id"]) - 1)
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
            record["ctx_utts_text"] = context_queries_and_answers
            # record["last_response"] = last_response
            record["pos_docs_text"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids
            record["neg_docs_text"] = []
            record["prepos_neg_docs_pids"] = list(context_pos_docs_pids - set(pos_docs_pids))
            f.write(json.dumps(record))
            f.write('\n')

            # last_response = positive_ctxs[0]["title"] + ". " + positive_ctxs[0]["text"]
            context_pos_docs_pids |= set(pos_docs_pids)
            context_queries_and_answers.append(query)
            context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])


    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    context_queries_and_answers = []
    with open(ouput_test_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                pos_docs.append(pos["title"] + ". " + pos["text"])
                pos_docs_pids.append(int(pos["passage_id"]) - 1)
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
            record["ctx_utts_text"] = context_queries_and_answers
            # record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids
            record["neg_docs"] = []
            record["prepos_neg_docs_pids"] = list(context_pos_docs_pids - set(pos_docs_pids))
            f.write(json.dumps(record))
            f.write('\n')

            # last_response = positive_ctxs[0]["title"] + ". " + positive_ctxs[0]["text"]
            context_pos_docs_pids |= set(pos_docs_pids)
            context_queries_and_answers.append(query)
            context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])



# tag: Train or Dev
def merge_rewrite_info(rewrite_file, orig_file, new_file, tag):
    with open(rewrite_file, "r") as f:
        data = json.load(f)
    
    sid2rewrite = {}
    for line in tqdm(data):
        sample_id = "{}_{}_{}".format("TopiOCQA-{}".format(tag), line["conv_id"], line["turn_id"])
        rewrite = line['question']
        sid2rewrite[sample_id] = rewrite

    with open(new_file, "w") as fw, open(orig_file, 'r') as fr:
        for line in tqdm(fr):
            line = json.loads(line)
            rewrite = sid2rewrite[line['sample_id']]
            line['oracle_utt_text'] = rewrite
            fw.write(json.dumps(line) + '\n')
            


if __name__ == "__main__":
    
    raw_train_file_path = "gold_train.json"
    raw_dev_file_path = "gold_dev.json"
    output_train_file_path = "train.json"
    ouput_test_file_path = "test.json"
    gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, ouput_test_file_path)

    raw_dev_file_path = "gold_dev.json"
    output_qrel_file_path = "topiocqa_qrel.tsv"
    gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path)

    rewrite_file = "rewrite_dev.json"
    orig_file = "test.json"
    new_file = "new_test.json"
    merge_rewrite_info(rewrite_file, orig_file, new_file, "Dev")

    