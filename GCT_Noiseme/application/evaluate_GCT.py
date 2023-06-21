import sys
import os, argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.data_generator import *
import framework.config as config
from framework.context_model import *
from framework.processing import *

import warnings

warnings.filterwarnings('ignore')


def main(argv):
    lr_decay = False
    epochs = 500
    config.batch_size = 64
    reverse_token_cues = False  # True
    start_end_labels = False  # True

    Encoder_layers = 4
    Decoder_layers = 4

    system_name = 'sys_b' + str(config.batch_size) + '_e' + str(epochs)

    if reverse_token_cues:
        system_name = system_name + '_reverse_token_cues'
    if start_end_labels:
        system_name = system_name + '_bibound'

    model_name = 'm_' + str(Encoder_layers) + '_' + str(Decoder_layers)

    if not lr_decay:
        suffix = model_name
    else:
        suffix = model_name + '_lrdecay'

    model_dir = os.path.join(os.getcwd(), system_name, suffix)

    generator = DataGenerator_Noiseme_2tokens(start_end_labels=start_end_labels, batch_size=config.batch_size)
    generator.label_set()

    model = Gated_cTransformer_clip(ntoken=generator.words_num,
                                    encoder_layers=Encoder_layers,
                                    decoder_layers=Decoder_layers)

    pred_event_seq_dir = model_dir + '_pred_seq'
    pred_event_at_dir = model_dir + '_pred_at'

    testing_model(generator, model, model_dir, pred_event_seq_dir, pred_event_at_dir,
                          reverse=reverse_token_cues, start_end_labels=start_end_labels)

    evaluate_audio_tagging(pred_event_at_dir)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

