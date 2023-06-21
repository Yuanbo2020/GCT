import sys
import os, argparse

gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.data_generator import *
from framework.context_model import *
from framework.processing import *

import warnings

warnings.filterwarnings('ignore')


def main(argv):
    epochs = 500
    config.batch_size = 64
    reverse_token_cues = False
    start_end_labels = False
    lr_decay = False

    Encoder_layers = 1
    Decoder_layers = 2

    system_name = 'sys_b' + str(config.batch_size) + '_e' + str(epochs)

    if reverse_token_cues:
        system_name = system_name + '_reverse_token_cues'
    if start_end_labels:
        system_name = system_name + '_bibound'

    model_name = 'm_' + str(Encoder_layers) + '_' + str(Decoder_layers)
    suffix = model_name

    if lr_decay:
        suffix = suffix + '_lrdecay'

    model_dir = os.path.join(os.getcwd(), system_name, suffix)

    generator = DataGenerator_Noiseme_2tokens(start_end_labels=start_end_labels, batch_size=config.batch_size)
    generator.label_set()

    # interspeech SAT
    model = Transformer_context(ntoken=generator.words_num,
                                encoder_layers=Encoder_layers,
                                decoder_layers=Decoder_layers)

    print(model)
    if config.cuda:
        model.cuda()

    training_process(generator, model, model_dir, epochs,
                                     testing=True,
                                     reverse=reverse_token_cues, start_end_labels=start_end_labels)

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

