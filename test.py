from Experiment import Experiment
from config import Config

config = Config()


def test():

    # Building the wrapper
    wrapper = Experiment(load_train=False)

    wrapper.test()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    test()
