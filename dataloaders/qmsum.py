import os
from dataloaders.unified_data import DialSumBase
from config import Config

config = Config()

MAP = {'train': 'train', 'valid': 'val', 'test': 'test'}

class QMSum(DialSumBase):
    """The QMSum dataset."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(QMSum, self).__init__(mode, retriever_tokenizer, generator_tokenizer)
        self.root = os.path.join('data', 'QMSum')

        # TODO: add qmsum_ before filename, see arxiv.py
        self.cached_features_file = os.path.join(self.root, '{}_cached_qmsum'.format(MAP[self.mode]))

        self.file_name = os.path.join(self.root, 'qmsum_{}_with_oracle.jsonl'.format(MAP[self.mode]))

        self.load_features_from_cache()

    def get_features(self):
        self.features = self.read_dialogue_summarization()
        print('QMSum data successfully read.')

