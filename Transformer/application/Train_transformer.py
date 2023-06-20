import sys
import os, argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.data_generator import *
from framework.context_model import *
from framework.processing import *

import warnings

warnings.filterwarnings('ignore')


def main(argv):
    lr_decay = False
    epochs = 500
    config.batch_size =20# 64
    reverse_token_cues = False  # True
    start_end_labels = False  # True

    Encoder_layers = 6
    Decoder_layers = 6

    system_name = 'sys_b' + str(config.batch_size) + '_e' + str(epochs)

    if reverse_token_cues:
        system_name = system_name + '_reverse_token_cues'
    if start_end_labels:
        system_name = system_name + '_bibound'

    model_name = 'm_' + str(Encoder_layers) + '_' + str(Decoder_layers)

    model_dir = os.path.join(os.getcwd(), system_name, model_name)

    generator = DataGenerator_Noiseme_2tokens(start_end_labels=start_end_labels, batch_size=config.batch_size)
    generator.label_set()

    model = Transformer(ntoken=generator.words_num, encoder_layers=Encoder_layers, decoder_layers=Decoder_layers)

    print(model)
    if config.cuda:
        model.cuda()

    training_process(generator, model, model_dir, epochs)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)



 