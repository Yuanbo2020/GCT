import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.process import *
import framework.config as config
from framework.model import *

import warnings

warnings.filterwarnings('ignore')


def main(argv):
    epochs = 500
    config.batch_size = 64

    system_name = 'sys_' + str(config.batch_size) + '_e' + str(epochs)

    model_name = 'm_gru_ctc'

    model_dir = os.path.join(os.getcwd(), system_name, model_name)
    ctc_frame_prob_dir = model_dir + '_frame_prob'
    ctc_at_dir = model_dir + '_at'

    using_model = model_icassp2019

    (sample_num, frame_num, height) = config.train_x.shape

    predict_ctc_model(model_dir, using_model, frame_num, height, ctc_frame_prob_dir, ctc_at_dir,
                                      only_final_epoch=epochs)



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

