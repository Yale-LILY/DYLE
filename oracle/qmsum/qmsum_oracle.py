import json
import os
from os.path import join
import torch
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import tempfile
import subprocess as sp
from datetime import timedelta
from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from pyrouge import Rouge155
from pyrouge.utils import log
from rouge import Rouge

# Change to own Rouge Path
_ROUGE_PATH = '.../rouge/ROUGE-1.5.5/'
rouge_diff_thresh = 0
test_pruning_thresh = 0
sent_limit = 300

# Evaluate ROUGE score given two directories
def eval_rouge(dec_dir, ref_dir):
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    cmd = '-c 95 -r 1000 -n 2 -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
            + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
            + cmd
            + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
        R_1 = float(output.split('\n')[3].split(' ')[3])
        R_2 = float(output.split('\n')[7].split(' ')[3])
        R_L = float(output.split('\n')[11].split(' ')[3])
        print(output)

# Compute average of ROUGE score using PyROUGE (faster)
def rouge(dec, ref):
    if dec == '' or ref == '':
        return 0.0
    rouge = Rouge()
    scores = rouge.get_scores(dec, ref)
    return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

# Get decoded sentence given index
def get_dec(text, idx):
    dec = []
    for i in idx:
        dec.append(text[i])
    return ' '.join(dec)

# Do first-round of filtering. Remove irrelevant snippets
def text_pruning(text, ref):
    new_text = []
    for i in range(len(text)):
        if not text[i] or text[i] == ".":
            continue
        try:
            cur_score = rouge(text[i], ref)
        except:
            continue
        if cur_score > test_pruning_thresh:
            new_text.append(text[i])
    return new_text

# Obtain the extractive oracle using greedy search
def get_oracle(text, ref):
    original_text = text
    text = text_pruning(text, ref)
    score = 0.0
    oracle_idx = []
    while True:
        best_score = 0.0
        best_idx = -1
        for i in range(len(text)):
            if i in oracle_idx:
                continue
            cur_idx = oracle_idx + [i]
            cur_idx.sort()
            dec = get_dec(text, cur_idx)
            cur_score = rouge(dec, ref)
            if cur_score > best_score:
                best_score = cur_score
                best_idx = i

        if best_score > score + rouge_diff_thresh:
            score = best_score
            oracle_idx += [best_idx]
            oracle_idx.sort()
        else:
            break

    original_indexes = []
    for index in oracle_idx:
        original_indexes.append(original_text.index(text[index]))

    return get_dec(text, oracle_idx), original_indexes


mode = "train"
# Change to own path
path = '.../QMSum/data/ALL/jsonl/{}.jsonl'.format(mode)
data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))

source_content = []
target_content = []

for data_entity in tqdm(data):

    text = []
    for j in range(len(data_entity['meeting_transcripts'])):
        text.append(' '.join(word_tokenize(data_entity['meeting_transcripts'][j]['content'].lower())))
    for j in range(len(data_entity['general_query_list'])):
        source_content.append(text)
        target_content.append(' '.join(word_tokenize(data_entity['general_query_list'][j]['answer'].lower())))
    for j in range(len(data_entity['specific_query_list'])):
        source_content.append(text)
        target_content.append(' '.join(word_tokenize(data_entity['specific_query_list'][j]['answer'].lower())))

print('{} total {}'.format(mode, len(target_content)))
assert(len(source_content) == len(target_content))

num_workers = 20
per_batch = len(source_content) // num_workers

def extract_oracle(samples):

    for idx, data_entity in samples:

        text = data_entity[0]
        ref = data_entity[1]

        oracle, indexes = get_oracle(text, ref)

        # Obtain oracle index
        with open('./greedy/ref_{}/{}.ref'.format(mode, idx), 'w') as f:
            for sent in sent_tokenize(ref):
                print(sent, file=f)
    
        # Obtain gold summary reference
        with open('./greedy/dec_{}/{}.dec'.format(mode, idx), 'w') as f:
            for sent in sent_tokenize(oracle):
                print(sent, file=f)

        # Obtain decoded oracle
        with open('./greedy/index_{}/{}.dec'.format(mode, idx), 'w') as f:
            print(indexes, file=f)

        print(indexes)
        print('{}: Finished {} pairs'.format(mode, idx))


with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    samples = []
    for idx, sample in enumerate(zip(source_content, target_content)):

        samples.append((idx, sample))
        if len(samples) % 3 == 0 or idx == len(source_content) - 1:

            futures.append(executor.submit(partial(extract_oracle, samples)))
            samples = []

    results = [future.result() for future in tqdm(futures)]

print('Start evaluating ROUGE score')
eval_rouge('./greedy/dec_{}'.format(mode), './greedy/ref_{}'.format(mode))