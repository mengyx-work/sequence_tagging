from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("tmp/model.weights/") # optional, restore weights

    # create datasets
    delimiter_ = '\t'
    dev   = CoNLLDataset(config.filename_dev, delimiter_, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, delimiter_, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
