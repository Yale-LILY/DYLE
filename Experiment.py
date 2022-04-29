import os
import numpy as np
import torch
import nltk
from dataloaders.qmsum import QMSum
from dataloaders.arxiv import Arxiv
from dataloaders.govreport import GovReport

import random
from tqdm import tqdm
from config import Config
from utils.utils import (gpu_wrapper, rouge_with_pyrouge)
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaForTokenClassification,
                          BartTokenizer,
                          AdamW)
from Modules.dynamic_rag import DynamicRagForGeneration
from nltk.tokenize import sent_tokenize, word_tokenize
config = Config()
ROUND = config.ROUND
EPSILON = 1e-10
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Experiment(object):

    def __init__(self, load_train=True):

        # Load retriever tokenizer.
        self.retriever_tokenizer = RobertaTokenizer.from_pretrained(config.retriever_name_or_path)

        # Load retriever model.
        self.retriever = RobertaForTokenClassification.from_pretrained(config.retriever_name_or_path,
                                                                       num_labels=1,
                                                                       gradient_checkpointing=True)

        # Load generator tokenizer.
        self.generator_tokenizer = BartTokenizer.from_pretrained(config.generator_name_or_path)

        # Load generator model.
        self.generator = DynamicRagForGeneration.from_pretrained(config.generator_name_or_path,
                                                                 n_docs=config.top_k,
                                                                 gradient_checkpointing=True)

        # Load loss.
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='none')

        self.modules = ['retriever', 'generator', 'criterion_cls']

        # Load dataset.
        print('----- Loading data -----')
        if config.target_task in ['qmsum-latent',
                                  ]:
            if load_train:
                self.train_set = QMSum('train', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
            self.val_set = QMSum('valid', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
            self.test_set = QMSum('test', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
        elif config.target_task in ['arxiv-latent',
                                    ]:
            if load_train:
                self.train_set = Arxiv('train', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
            self.val_set = Arxiv('valid', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
            self.test_set = Arxiv('test', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
        elif config.target_task in ["govreport-latent",
                                    ]:
            if load_train:
                self.train_set = GovReport('train', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
            self.val_set = GovReport('valid', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)
            self.test_set = GovReport('test', retriever_tokenizer=self.retriever_tokenizer, generator_tokenizer=self.generator_tokenizer)      
        else:
            raise ValueError()

        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            if getattr(self, module) is not None:
                setattr(self, module, gpu_wrapper(getattr(self, module)))

        self.scopes = {'cls': ['retriever'], 'gen': ['generator']}
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.iter_num = 0
        self.best_metric = - float('inf')
        self.decay_num = 0
        self.no_improvement = 0

        # Tokenization for BLEU.
        nltk_wordpunk_tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.bleu_tokenizer = lambda x: nltk_wordpunk_tokenizer.tokenize(x)

    def restore_model(self, modules, dirs=None):
        print('Loading the trained best models...')
        if dirs is not None:
            assert len(modules) == len(dirs)
            for module, directory in zip(modules, dirs):
                path = os.path.join(directory, 'best-{}.ckpt'.format(module))
                getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                      strict=True)
        else:
            for module in modules:
                path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
                getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                      strict=True)

    def save_step(self, modules):
        for module in modules:
            path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
            torch.save(getattr(self, module).state_dict(), path)
        print('Saved model checkpoints into {}...\n\n\n\n\n\n\n\n\n\n\n\n'.format(config.save_model_dir))

    def zero_grad(self):
        for scope in self.scopes:
            getattr(self, scope + '_optim').zero_grad()

    def step(self, scopes):
        if config.max_grad_norm is not None:
            grouped_params = []
            for scope in scopes:
                grouped_params.extend(getattr(self, scope + '_grouped_parameters'))

            clip_grad_norm_(grouped_params, config.max_grad_norm)

        for scope in scopes:
            # Optimize.
            getattr(self, scope + '_optim').step()

    def update_lr_by_half(self):
        self.decay_num += 1
        for scope in self.scopes:
            setattr(self, scope + '_lr', getattr(self, scope + '_lr') / 2)  # Half the learning rate.
            for param_group in getattr(self, scope + '_optim').param_groups:
                param_group['lr'] = getattr(self, scope + '_lr')
            print('{}: {}'.format(scope + '_lr', getattr(self, scope + '_lr')))

    def set_requires_grad(self, modules, requires_grad):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = requires_grad

    def set_training(self, mode):
        for module in self.modules:
            if getattr(self, module) is not None:
                getattr(self, module).train(mode=mode)

    def train(self):
            
        self.build_optim()

        # Train.
        epoch = 0
        self.zero_grad()
        while True:
            self.train_epoch(epoch)
            epoch += 1
            if self.decay_num >= config.max_decay_num:
                break

        # Test.
        self.test()

    def build_optim(self):
        # Set trainable parameters, according to the frozen parameter list.
        for scope in self.scopes.keys():
            optimizer_grouped_parameters = [
                {'params': [],
                 'weight_decay': config.weight_decay},
                {'params': [],
                 'weight_decay': 0.0},
            ]
            no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

            for module in self.scopes[scope]:
                if getattr(self, module) is not None:
                    for n, p in getattr(self, module).named_parameters():
                        # k is the parameter name; v is the parameter value.
                        if p.requires_grad:
                            # Weight decay.
                            if not any(nd in n for nd in no_decay):
                                print("[{} Trainable:]".format(module), n)
                                optimizer_grouped_parameters[0]['params'].append(p)
                            else:
                                print("[{} Trainable (bias/LN):]".format(module), n)
                                optimizer_grouped_parameters[1]['params'].append(p)
                        else:
                            print("[{} Frozen:]".format(module), n)

            if config.optimizer == 'adam':
                setattr(self, scope + '_optim', AdamW(optimizer_grouped_parameters, lr=getattr(self, scope + '_lr')))
            else:
                raise ValueError()

            setattr(self,
                    scope + '_grouped_parameters',
                    optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params'])

    def test(self):
        self.restore_model(['retriever', 'generator'])

        # Evaluate.
        if config.target_task in ['qmsum-latent',
                                  'arxiv-latent',
                                  'govreport-latent',
                                  ]:
            if config.target_task in ['govreport-latent']:
                beam_size = 4  # Use beam_size = 1 for validation
            elif config.target_task in ['arxiv-latent']:
                beam_size = 5
            elif config.target_task in ['qmsum']:
                beam_size = 6
            else:
                beam_size = 1
            self.seq_evaluate_gen(test=True, beam_size=beam_size)
        else:
            raise ValueError()

    def train_epoch(self, epoch_id):

        train_dataloader = DataLoader(self.train_set,
                                      batch_size=config.train_batch_size // config.gradient_accumulation_steps,
                                      shuffle=True,
                                      num_workers=config.num_workers)

        for data in train_dataloader:
            self.iter_num += 1
            self.set_training(mode=True)

            if config.target_task in ['qmsum-latent',
                                      'arxiv-latent',
                                      'govreport-latent',
                                      ]:
                # Process data.
                data = self.cuda_data(*data)

                retriever_input_ids, global_attention_mask, cls_ids, oracle, \
                    context_input_ids, context_attention_mask, labels, index = data

                assert retriever_input_ids.shape[0] == 1

                num_oracle = len(oracle[0])
                # Forward.
                retriever_outputs = self.retriever(input_ids=retriever_input_ids.squeeze(0),
                                                   output_hidden_states=True)
                retriever_all_logits = retriever_outputs.logits
                retriever_all_logits = retriever_all_logits.squeeze(2)
                retriever_cls_logits = retriever_all_logits.contiguous().view(-1)[cls_ids.squeeze(0).cpu().tolist()].unsqueeze(0)

                # Retrieval loss.
                ret_loss = 0
                for turn_id in oracle[0].cpu().tolist():
                    ret_loss = ret_loss + self.criterion_cls(input=retriever_cls_logits,
                                                             target=gpu_wrapper(torch.LongTensor([turn_id])))

                if num_oracle > 0:
                    ret_loss = ret_loss / num_oracle

                # Generation loss.
                if config.loss_alpha != 0 or num_oracle == 0:

                    if oracle.shape[1] != 0:
                        k = min(config.top_k, retriever_cls_logits.shape[1])
                        retriever_topk_indices = oracle.squeeze(0).cpu().tolist()[:k]
                        if config.hybrid_train and len(retriever_topk_indices) < k:
                            _, real_retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                            real_retriever_topk_indices = real_retriever_topk_indices[0].cpu().tolist()
                            retriever_topk_indices = retriever_topk_indices + [idx for idx in real_retriever_topk_indices if idx not in retriever_topk_indices]
                            retriever_topk_indices = retriever_topk_indices[:k]
                        doc_scores = retriever_cls_logits[:, retriever_topk_indices]
                    else:
                        doc_scores, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                        retriever_topk_indices = retriever_topk_indices[0].cpu().tolist()

                    if len(retriever_topk_indices) < config.top_k:
                        doc_scores = torch.cat([doc_scores, gpu_wrapper(torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                        retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (config.top_k - len(retriever_topk_indices))

                    generator_outputs = self.generator(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                       context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                       doc_scores=doc_scores,
                                                       labels=labels)
                    seq_loss = generator_outputs.loss
                    consistency_loss = generator_outputs.consistency_loss

                else:
                    seq_loss = 0

                tot_loss = seq_loss * config.loss_alpha + ret_loss

                tot_loss = tot_loss + config.consistency_alpha * consistency_loss

                # Backward.
                if config.gradient_accumulation_steps > 1:
                    tot_loss = tot_loss / config.gradient_accumulation_steps

                tot_loss.backward()

                if self.iter_num % config.gradient_accumulation_steps == 0:
                    # ----- Backward for scopes: ['cls', 'gen'] -----
                    self.step(['cls', 'gen'])

                    self.zero_grad()

            elif config.target_task in ['qmsum-oracle',
                                        ]:
                raise NotImplementedError()
            else:
                raise ValueError()

            # Evaluation.
            if self.iter_num % (config.save_steps * config.gradient_accumulation_steps) == 0:
                if config.target_task in ['govreport-latent', 'arxiv-latent']:
                    beam_size = 1  # Use beam_size = 1 for validation
                else:
                    beam_size = 5
                no_improvement = self.seq_evaluate_gen(test=False, beam_size=beam_size)

                # Learning rate decay.
                if no_improvement and self.iter_num > config.start_decay:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0

                if self.no_improvement == config.no_improvement_decay:
                    self.update_lr_by_half()
                    self.no_improvement = 0

    def seq_evaluate_gen(self, test, beam_size):
        self.set_training(mode=False)

        print('beam_size = {}'.format(beam_size))

        if test:
            the_set = self.test_set
        else:
            the_set = self.val_set
        eval_dataloader = DataLoader(the_set, batch_size=1, shuffle=False, num_workers=config.num_workers)

        # Eval!
        print("\n\n\n\n***** Running evaluation *****")
        print("  Num examples = {}".format(len(the_set)))
        print("  Batch size = {}".format(1))

        predictions = []
        topks = []
        doc_scoreses = []

        top_5 = [False, True][0]
        tot = 0

        for data in tqdm(eval_dataloader):
            tot += 1
            if tot > 3 and top_5:
                break
            # Process data.
            data = self.cuda_data(*data)
            retriever_input_ids, global_attention_mask, cls_ids, oracle, \
                context_input_ids, context_attention_mask, labels, index = data

            assert retriever_input_ids.shape[0] == 1

            # Forward (prediction).
            with torch.no_grad():
                retriever_outputs = self.retriever(input_ids=retriever_input_ids.squeeze(0),
                                                   output_hidden_states=True)
                retriever_all_logits = retriever_outputs.logits
                retriever_all_logits = retriever_all_logits.squeeze(2)
                retriever_cls_logits = retriever_all_logits.contiguous().view(-1)[cls_ids.squeeze(0).cpu().tolist()].unsqueeze(0)  # shape = (1, n_turns)

                if config.oracle_test and oracle.shape[1] != 0:
                    k = min(config.top_k, retriever_cls_logits.shape[1])
                    doc_scores = torch.gather(retriever_cls_logits, dim=1, index=oracle)[:, :k]
                    retriever_topk_indices = oracle.squeeze(0).cpu().tolist()[:k]
                else:
                    doc_scores, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                    retriever_topk_indices = retriever_topk_indices[0].cpu().tolist()

                if len(retriever_topk_indices) < config.top_k:
                    doc_scores = torch.cat([doc_scores, gpu_wrapper(torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                    retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (config.top_k - len(retriever_topk_indices))

                if config.loss_alpha != 0:
                    outputs = self.generator.generate(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                      context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                      doc_scores=doc_scores,
                                                      num_beams=beam_size,
                                                      min_length=config.min_length,
                                                      max_length=config.max_target_len,
                                                      no_repeat_ngram_size=config.no_repeat_ngram_size,
                                                      length_penalty=config.length_penalty,
                                                      )
                    assert isinstance(outputs, torch.Tensor)
                    assert outputs.shape[0] == 1

                    # Predictions:
                    decoded_pred = self.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    cleaned_prediction = ["\n".join(sent_tokenize(" ".join(word_tokenize(pred)))) for pred in decoded_pred]
                else:
                    cleaned_prediction = ["empty prediction because loss_alpha = 0."]
                predictions.extend(cleaned_prediction)

                # top_k:
                decoded_topk = self.generator_tokenizer.batch_decode(context_input_ids[:, retriever_topk_indices].contiguous().view(config.top_k, -1), skip_special_tokens=True)
                if config.target_task in ['govreport-latent', "arxiv-latent"]:
                    cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                    topks.append(cleaned_topk)

                else:
                    cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent[sent.index(':') + 1:sent.index(' // ') if ' // ' in sent else len(sent)] for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                    topks.append(cleaned_topk)

                doc_scoreses.append(doc_scores[0])

        # Load references.
        references = ["\n".join(sent_tokenize(" ".join(word_tokenize(sent)))) for sent in the_set.get_references()]

        # ROUGE.
        rouge1, rouge2, rougeL = rouge_with_pyrouge(preds=predictions, refs=references)
        print(rouge1, rouge2, rougeL)

        rouge1_topk, rouge2_topk, rougeL_topk = rouge_with_pyrouge(preds=topks, refs=references)

        print(rouge1_topk, rouge2_topk, rougeL_topk)

        if config.loss_alpha != 0:
            metric = rouge1 + rouge2 + rougeL
        else:
            metric = rouge1_topk + rouge2_topk + rougeL_topk

        if not test and metric > self.best_metric:
            self.best_metric = metric
            self.save_step(['retriever', 'generator'])

            peep_num = 3
            for sent_id in range(peep_num):
                print('Pred:\n{}'.format(predictions[sent_id]))
                print('-' * 20)
                print('topk:\n{}'.format(topks[sent_id]))
                print('-' * 20)
                print('Ref:\n{}'.format(references[sent_id]))
                print('-' * 20)
                print()
                print('=' * 50)

        self.set_training(mode=True)

        base_name = '{}.gen'.format('test' if test else 'valid')
        save_path = os.path.join(config.sample_dir, base_name)
        torch.save((predictions, references), save_path)

    def number_parameters(self):
        print('Number of retriever parameters', sum(p.numel() for p in self.retriever.parameters()))
        print('Number of generator parameters', sum(p.numel() for p in self.generator.parameters()))

    @staticmethod
    def cuda_data(*data, **kwargs):
        if len(data) == 0:
            raise ValueError()
        elif len(data) == 1:
            return gpu_wrapper(data[0], **kwargs)
        else:
            return [gpu_wrapper(item, **kwargs) for item in data]


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    # print(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm