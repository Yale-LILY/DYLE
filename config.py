import os
import torch
import glob


class Config(object):

    def __init__(self):

        self.target_task = ['qmsum-latent',  # 0: qmsum-latent
                            'arxiv-latent',  # 1: arxiv-latent
                            'govreport-latent',  # 2: govreport-latent
                            ][2]  

        self.retriever = ['roberta',
                          ][0] 
        self.retriever_name_or_path = {'roberta': 'roberta-base',
                                       }[self.retriever]
        self.generator = ['dynamic-rag',
                          ][0] 
        self.generator_name_or_path = {'dynamic-rag': 'facebook/bart-large',
                                       }[self.generator]

        # Training configuration.
        self.max_grad_norm = 1.0
        self.cls_lr = 5e-5 
        self.gen_lr = 5e-5 

        self.overwrite_cache = False 
        self.weight_decay = 0.0  

        self.start_decay = 0
        self.max_decay_num = 3
        self.no_improvement_decay = 5
        self.optimizer = 'adam'
        self.filtered_oracle = False
        
        self.early_preprocess = True

        self.train_batch_size = 8 
        self.eval_batch_size = 1
        self.test_batch_size = 1
        self.gradient_accumulation_steps = 8 
        assert self.train_batch_size % self.gradient_accumulation_steps == 0

        # Miscellaneous.
        self.num_workers = 8 
        self.ROUND = 4
        self.seed = [0, 1, 2, 3, 4][0] 
        self.gpu = torch.cuda.is_available()

        # Method-related.
        if self.retriever == 'roberta':
            self.max_retrieval_len = 512
            self.max_chunks = 50
        else:
            raise NotImplementedError()

        if self.target_task in ['qmsum-latent',
                                ]:
            self.use_oracle = True
            self.use_query = True
            self.gen_lr = 1e-6  

            self.oracle_type = ['greedy',][0] 
            self.oracle_train = [False, True][1] 
            if self.oracle_train:
                self.hybrid_train = [False, True][1]  
            self.oracle_test = [False, True][0] 
            self.loss_alpha = [0, 0.05, 0.1, 1][3]  
            self.window_size = 0 
            self.top_k = 20  
            self.min_length = 100 
            self.no_repeat_ngram_size = 2 
            self.max_source_len = 300 
            self.max_target_len = 600 

            self.consistency_alpha = [0, 1, 2, 3, 5, 10][1] 
            self.detach_generator_consistency = [False, True][1] 
            self.length_penalty = 1

            self.save_steps = 100
            
        elif self.target_task in ['arxiv-latent',
                                  ]:
            self.use_oracle = True
            self.use_query = False
            self.early_preprocess = False

            self.oracle_type = ['greedy', ][0] 
            self.oracle_train = [False, True][1]  
            if self.oracle_train:
                self.hybrid_train = [False, True][1] 
            self.oracle_test = [False, True][0] 
            self.loss_alpha = [0, 0.1, 0.5, 1, 5][2]  
            self.window_size = 0 
            self.top_k = 25  
            self.min_length = 150  
            self.no_repeat_ngram_size = 3 

            self.max_source_len = 64  
            self.max_target_len = 900  

            self.consistency_alpha = [0, 1, 2, 3, 5, 10, 15][5] 
            self.detach_generator_consistency = [False, True][1] 
            self.length_penalty = 1
            self.save_steps = 500 
            
        elif self.target_task in ['govreport-latent',
                                  ]:
            self.use_oracle = True
            self.use_query = False

            self.oracle_type = ['greedy',][0] 
            self.oracle_train = [False, True][1]
            if self.oracle_train:
                self.hybrid_train = [False, True][1] 
            self.oracle_test = [False, True][0]
            self.loss_alpha = [0, 0.1, 0.5, 1, 5][2]
            self.window_size = 0 
            self.top_k = 25
            self.min_length = 500
            self.no_repeat_ngram_size = 5 
            self.max_source_len = 64
            self.max_target_len = 900

            self.consistency_alpha = [0, 0.1, 1, 2, 3, 5, 10][2] 
            self.detach_generator_consistency = [False, True][1]
            self.length_penalty = 2.0 

            self.save_steps = 500 
            self.retriever_save_steps = 1000
        else:
            raise ValueError()

        # Directories.
        self.log_dir = self.model_specific_dir('outputs/logs')
        remove_all_under(self.log_dir)
        self.save_model_dir = self.model_specific_dir('outputs/saved_model')
        self.sample_dir = self.model_specific_dir('outputs/sampled_results')
        self.tmp_dir = self.model_specific_dir('outputs/temp_results')

    def model_specific_dir(self, root):
        """ model-normalization """
        directory = {
            'qmsum-latent': 'QMSum-DYLE',
            'arxiv-latent': 'ArXiv-DYLE',
            'govreport-latent': 'GovReport-DYLE',
        }[self.target_task]

        ret = os.path.join(root, directory)
        if not os.path.exists(ret):
            os.mkdir(ret)
        return ret


def remove_all_under(directory):
    for file in glob.glob(os.path.join(directory, '*')):
        os.remove(file)