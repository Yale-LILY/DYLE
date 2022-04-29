import os
import shutil
from Experiment import Experiment
from config import Config

config = Config()


def test():

    # Building the wrapper
    wrapper = Experiment()

    wrapper.number_parameters()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    test()
