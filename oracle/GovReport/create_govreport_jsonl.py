import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import os.path

mode = "train"
# Download the GovReport datasets and set the path
source_file = ".../govreport/{}.source".format(mode)
target_file = ".../govreport/{}.target".format(mode)
sent_limit = 64

with open(source_file) as f:
    source_content = f.readlines()

source_content = [x.strip() for x in source_content] 

with open(target_file) as f:
    target_content = f.readlines()

target_content = [x.strip() for x in target_content] 

data = []
idx = 0

def process_article_sent_tokenize(article):

    article = " ".join(word_tokenize(article.lower()))
    article = sent_tokenize(article)
    return article

def insert_new(article_list, sent):

    token_list = word_tokenize(sent) 
    article_list.append(" ".join(token_list[:sent_limit]))

    if len(token_list) > sent_limit:
        insert_new(article_list, " ".join(token_list[sent_limit:]))

def process_article(article):

    article = process_article_sent_tokenize(article)
    new_article = []

    for sent in article:
        insert_new(new_article, sent)

    return new_article
        
for report, summary in tqdm(zip(source_content, target_content), total = len(source_content)):

    report = process_article(report)
    summary = " ".join(word_tokenize(summary.lower()))
    json_entry = {}
    json_entry["report"] = report
    json_entry["summary"] = summary
    # Change to own path
    fname = ".../GovReport/GovReport_new/index_{}/{}.dec".format(mode, idx)

    with open(fname) as f:
        idx += 1
        oracle = f.read().strip()
        json_entry["oracle"] = oracle
        if len(oracle) <= 2:
            print(idx, oracle)
        else:    
            data.append(json_entry)

# Change to own path
with open('.../Govreport/govreport_{}.jsonl'.format(mode), 'w') as outfile:
    for entry in data:
        json.dump(entry, outfile)
        outfile.write('\n')

