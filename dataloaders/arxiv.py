import os
from dataloaders.unified_data import PaperSumBase
from config import Config

config = Config()

MAP = {'train': 'train', 'valid': 'val', 'test': 'test'}


class Arxiv(PaperSumBase):
    """The Arxiv dataset."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(Arxiv, self).__init__(mode, retriever_tokenizer, generator_tokenizer)

        self.root = os.path.join('data', 'arxiv')

        self.cached_features_file = os.path.join(self.root, '{}_cached_arxiv'.format(MAP[mode]))

        self.file_name = "oracle/arxiv/index_train/"

        self.load_features_from_cache()

    def get_features(self):
        self.features = self.read_paper_summarization()
        print('ArXiv data successfully read.')

