import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import os.path
from datasets import load_dataset

mode = "test"

dataset = load_dataset("scientific_papers", "arxiv", split=mode)
source_content = [data_entity["article"] for data_entity in dataset] 
target_content = [data_entity["abstract"] for data_entity in dataset] 

data = []
idx = 0
token_limit = 64

def insert_new(article_list, sent):

    token_list = word_tokenize(sent) 

    while len(token_list) > token_limit:

        article_list.append(" ".join(token_list[:token_limit]))
        token_list = token_list[token_limit:]
    
    article_list.append(" ".join(token_list))

def process_article_sent_tokenize(article):

    article = " ".join(word_tokenize(article.lower()))
    article = sent_tokenize(article)
    return article

def process_article(article):

    article = process_article_sent_tokenize(article)
    new_article = []

    for sent in article:
        insert_new(new_article, sent)

    return new_article

for article, summary in tqdm(zip(source_content, target_content), total = len(source_content)):

    json_entry = {}
    json_entry["article"] = article
    json_entry["summary"] = summary
    fname = "./index/{}.dec".format(mode, idx)

    with open(fname) as f:
        idx += 1
        oracle = f.read().strip()
        oracle = json.loads(oracle)        

        if len(oracle) > 100:
            article = process_article(article)
