from torch.utils import data
import torch
import os
import re
import json
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
from utils.clean_utils import tokenize, clean_data
from config import Config
from datasets import load_dataset
from os import path
config = Config()


class DatasetBase(data.Dataset):
    """The base dataset for Text."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(DatasetBase, self).__init__()
        self.mode = mode
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer
        assert self.mode in ['train', 'valid', 'test']

        self.cached_features_file = ''
        self.features = []

    def tokenize_retriever(self, text, query, oracle):
        # Tokenize.
        tok_query = self.retriever_tokenizer(query).input_ids + [self.retriever_tokenizer.sep_token_id]
        tok_text = [self.retriever_tokenizer(turn).input_ids + [self.retriever_tokenizer.sep_token_id] for turn in text]

        input_ids_list = []
        global_attention_mask_list = []
        cls_ids = []
        idx_offset = 0
        turn_id = 0
        list_id = 0
        while turn_id < len(tok_text) and list_id < config.max_chunks:
            # Init.
            input_ids = []
            global_attention_mask = []

            # Query.
            input_ids.extend(tok_query)
            global_attention_mask.extend([1] * len(tok_query))

            # text.
            while turn_id < len(tok_text):
                tok_turn = tok_text[turn_id]
                if len(input_ids) + len(tok_turn) > config.max_retrieval_len:
                    if len(input_ids) == len(tok_query):
                        tok_turn = tok_turn[:config.max_retrieval_len - len(input_ids)]
                    else:
                        break
                input_ids.extend(tok_turn)
                global_attention_mask.extend([0] * (len(tok_turn) - 1) + [1])
                cls_ids.append(len(input_ids) - 1 + idx_offset)
                turn_id += 1

            # Pad.
            assert len(input_ids) == len(global_attention_mask)
            num_pad = config.max_retrieval_len - len(input_ids)
            input_ids.extend([self.retriever_tokenizer.pad_token_id] * num_pad)
            global_attention_mask.extend([0] * num_pad)

            # Save.
            assert len(input_ids) == len(global_attention_mask) == config.max_retrieval_len
            input_ids_list.append(input_ids)
            global_attention_mask_list.append(global_attention_mask)
            idx_offset += config.max_retrieval_len
            list_id += 1

        oracles = [oracle_id for oracle_id in oracle if oracle_id < turn_id and len(text[oracle_id].split(" ")) > 3]

        retriever_inputs = {'input_ids': input_ids_list,
                            'global_attention_mask': global_attention_mask_list,
                            'cls_ids': cls_ids,
                            'oracle': oracles,
                            }
        return retriever_inputs

    def tokenize_generator(self, text, query, summary):
        context_input_ids = []
        labels = None
        padded_text = [''] * config.window_size + text + [''] * config.window_size
        context_attention_mask = []
        for turn_id in range(len(text)):
            # turns with a window size.
            contextualized_turn = self.generator_tokenizer.eos_token.join(padded_text[turn_id: turn_id + 1 + 2 * config.window_size])

            input_dict = self.generator_tokenizer.prepare_seq2seq_batch(src_texts=contextualized_turn + " // " + query,
                                                                        tgt_texts=summary,
                                                                        max_length=config.max_source_len,
                                                                        max_target_length=config.max_target_len,
                                                                        padding="max_length",
                                                                        truncation=True,
                                                                        )
            context_attention_mask.append(input_dict.attention_mask)
            context_input_ids.append(input_dict.input_ids)
            if labels is None:
                labels = input_dict.labels
            else:
                assert labels == input_dict.labels, '{} != {}'.format(labels, input_dict.labels)

        generator_inputs = {'context_input_ids': context_input_ids,
                            'context_attention_mask': context_attention_mask,
                            'labels': labels}

        if labels is None:
            raise ValueError(text)

        return generator_inputs

    def get_features(self):
        raise NotImplementedError()

    def load_features_from_cache(self):

        if not config.early_preprocess:
            self.cached_features_file = self.cached_features_file + '_late_preprocess'
        print("cached feature file address", self.cached_features_file)
        if os.path.exists(self.cached_features_file) and not config.overwrite_cache:
            print("Loading features from cached file {}".format(self.cached_features_file))
            self.features = torch.load(self.cached_features_file)
        else:
            self.get_features()
            print("Saving features into cached file {}".format(self.cached_features_file))
            torch.save(self.features, self.cached_features_file)

    def preprocess(self, session):
        raise NotImplementedError()

    def __getitem__(self, index):
        if config.early_preprocess:
            retriever_inputs, generator_inputs = self.features[index]
        else:
            session = self.features[index]
            retriever_inputs, generator_inputs = self.preprocess(session)

        retriever_input_ids = torch.LongTensor(retriever_inputs['input_ids'])
        global_attention_mask = torch.LongTensor(retriever_inputs['global_attention_mask'])
        cls_ids = torch.LongTensor(retriever_inputs['cls_ids'])
        oracle = torch.LongTensor(retriever_inputs['oracle'])

        context_input_ids = torch.LongTensor(generator_inputs['context_input_ids'])
        context_attention_mask = torch.LongTensor(generator_inputs['context_attention_mask'])
        labels = torch.LongTensor(generator_inputs['labels'])[:config.max_target_len]

        return retriever_input_ids, global_attention_mask, cls_ids, oracle, \
            context_input_ids, context_attention_mask, labels, index

    def __len__(self):
        return len(self.features)


class DialSumBase(DatasetBase):
    """The base dataset for dialogue summarization."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(DialSumBase, self).__init__(mode, retriever_tokenizer, generator_tokenizer)
        self.file_name = 'file_name_not_specified'

    def get_references(self):
        with open(self.file_name) as f:
            references = []
            for line in tqdm(f.readlines()):
                # Process raw text.
                session = json.loads(line.strip())
                for pair in session['general_query_list']:
                    references.append(tokenize(pair['answer']))
                for pair in session['specific_query_list']:
                    references.append(tokenize(pair['answer']))
        return references

    def read_dialogue_summarization(self):
        print(("Reading dialogue as turns from {}".format(self.file_name)))
        with open(self.file_name) as f:
            features = []
            for line in tqdm(f.readlines()):
                # Process raw text.
                session = json.loads(line.strip())
                dialogue = [clean_data(turn['speaker'].lower() + ': ' + tokenize(turn['content']))
                            for turn in session['meeting_transcripts']]  # List of strings.
                queries = []  # List of strings.
                summaries = []  # List of strings.
                oracles = []  # List of index.
                for pair in session['general_query_list']:
                    queries.append(clean_data(tokenize(pair['query'])))
                    # '<s> ' + query + ' </s> ' + meeting + ' </s>'
                    summaries.append(tokenize(pair['answer']))
                    if '{}_oracle_idx'.format(config.oracle_type) in pair:
                        oracles.append(pair['{}_oracle_idx'.format(config.oracle_type)])
                    else:
                        oracles.append([0])
                for pair in session['specific_query_list']:
                    queries.append(clean_data(tokenize(pair['query'])))
                    summaries.append(tokenize(pair['answer']))
                    if '{}_oracle_idx'.format(config.oracle_type) in pair:
                        oracles.append(pair['{}_oracle_idx'.format(config.oracle_type)])
                    else:
                        oracles.append([0])

                # For retriever.
                assert len(queries) == len(summaries) == len(oracles)
                for query, summary, oracle in zip(queries, summaries, oracles):
                    retriever_inputs = self.tokenize_retriever(text=dialogue, query=query, oracle=oracle)

                    # For generator.
                    generator_inputs = self.tokenize_generator(text=dialogue, query=query, summary=summary)

                    features.append((retriever_inputs, generator_inputs))

            return features


class PaperSumBase(DatasetBase):
    """The base dataset for paper summarization."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(PaperSumBase, self).__init__(mode, retriever_tokenizer, generator_tokenizer)
        self.file_name = 'file_name_not_specified'
        self.sent_limit = 64
        if self.mode == "valid":
            self.dataset = load_dataset("scientific_papers", "arxiv", split="validation")
        else:
            self.dataset = load_dataset("scientific_papers", "arxiv", split=self.mode)

    def insert_new(self, article_list, sent):

        token_list = word_tokenize(sent) 

        while len(token_list) > self.sent_limit:

            article_list.append(" ".join(token_list[:self.sent_limit]))
            token_list = token_list[self.sent_limit:]
        
        article_list.append(" ".join(token_list))

    def process_article(self, article):

        new_article = []

        for sent in article:
            self.insert_new(new_article, sent)

        return new_article

    def get_references(self):

        references = []
        for session in tqdm(self.dataset):
            abstract = session["abstract"]
            references.append(tokenize(abstract))

        return references

    def preprocess(self, inputs):
        session, idx = inputs

        paper = sent_tokenize(tokenize(session["article"]))
        paper = self.process_article(paper)
        summary = tokenize(session["abstract"])

        if config.use_query:
            query = tokenize(session["title"])
        else:
            query = ""

        if self.mode == "train" or self.mode == "test":
            oracle_file_name = self.file_name + '{}.dec'.format(idx)
            if not path.exists(oracle_file_name):
                oracle = []
            else:
                with open(oracle_file_name) as f:

                    oracle = f.read().strip()
                    if len(oracle) <= 2:
                        oracle = []
                    else:
                        oracle = [int(x.strip()) for x in oracle[1:-1].split(",")]
        else:
            oracle = []

        retriever_inputs = self.tokenize_retriever(text=paper, query=query, oracle=oracle)

        if retriever_inputs is None:
            raise ValueError(paper, query, oracle)

        generator_inputs = self.tokenize_generator(text=paper, query=query, summary=summary)

        return retriever_inputs, generator_inputs

    def read_paper_summarization(self):

        print("Reading papers as sentences. Reading oracles from {}".format(self.file_name))
        features = []

        if config.use_oracle:
            idx = 0

            for session in tqdm(self.dataset):

                oracle_file_name = self.file_name + '{}.dec'.format(idx)

                if config.early_preprocess and path.exists(oracle_file_name):
                    features.append(self.preprocess(inputs=(session, idx)))
                else:

                    if len(session["abstract"]) > 3 and len(session["article"]) > 3:
                        features.append((session, idx))

                idx += 1
        else:
            raise NotImplementedError()

        return features


class ReportSumBase(DatasetBase):
    """The base dataset for GovReport summarization."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(ReportSumBase, self).__init__(mode, retriever_tokenizer, generator_tokenizer)
        self.file_name = 'file_name_not_specified'

    def get_references(self):
        with open(self.file_name) as f:
            references = []
            for line in tqdm(f.readlines()):
                # Process raw text.
                session = json.loads(line.strip())

                report = session["report"]
                summary = session["summary"]

                if len(report) < 10:
                    continue

                references.append(summary)
        return references

    def preprocess(self, session):

        # Already tokenized
        report = session["report"]
        summary = session["summary"]

        if len(report) < 10:
            return None
        if config.use_query:
            query = tokenize(session["query"])
        else:
            query = ""

        if config.use_oracle:
            oracle = session["oracle"]
            oracle = [int(x.strip()) for x in oracle[1:-1].split(",")]
        else:
            oracle = [0]

        retriever_inputs = self.tokenize_retriever(text=report, query=query, oracle=oracle)
        generator_inputs = self.tokenize_generator(text=report, query=query, summary=summary)

        return retriever_inputs, generator_inputs

    def read_report_summarization(self):

        print(("Reading gov report as turns from {}".format(self.file_name)))

        features = []

        with open(self.file_name) as f:
            for line in tqdm(f.readlines()):

                session = json.loads(line.strip())

                if config.early_preprocess:
                    feature = self.preprocess(session)
                    if feature is not None:
                        features.append(feature)
                else:
                    features.append(session)

        return features