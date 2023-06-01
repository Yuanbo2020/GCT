import framework.config as config
from framework.model import *

import os, pickle, time
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def training(model_dir, sample_num, conv_shape, model, epoch=100, batch=64):

    train_x = [config.train_x, config.train_y, np.ones(sample_num) * int(conv_shape[1]),
               np.array(config.train_truth_label_len)]
    train_y = np.ones(sample_num)

    sample_num = config.validation_x.shape[0]
    validation_x = [config.validation_x, config.validation_y, np.ones(sample_num) * int(conv_shape[1]),
                    np.array(config.val_truth_label_len)]
    validation_y = np.ones(sample_num)

    create_folder(model_dir)
    filepath = os.path.join(model_dir, "model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = Yuanbo_ModelCheckpoint(filepath=filepath, verbose=0)

    hist = model.fit(x=train_x, y=train_y,
                     batch_size=batch,
                     epochs=epoch,
                     validation_data=(validation_x, validation_y),
                     verbose=1, shuffle=True, callbacks=[save_model])

    print('hist.history:', hist.history)

    log_file = os.path.join(model_dir, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(str(hist.history))


def predict_ctc_model(model_dir, using_model, frame_num, height, ctc_frame_prob_dir, ctc_at_dir,
                      only_final=False, only_final_epoch=None):
    create_folder(ctc_at_dir)

    if not only_final:
        only_final_epoch = None

    weak_label_set = config.weak_label_set
    test_id = config.test_id

    from keras import backend as K

    model_path_list = []
    sub_model_list = []
    for sub_model in os.listdir(model_dir):
        if sub_model.endswith('.hdf5'):
            model_path = os.path.join(model_dir, sub_model)
            if only_final_epoch is None:
                model_path_list.append(model_path)
                sub_model_list.append(sub_model)
            elif str(only_final_epoch) in sub_model:
                model_path_list.append(model_path)
                sub_model_list.append(sub_model)

    for model_path, sub_model in zip(model_path_list, sub_model_list):

        at_file_path = os.path.join(ctc_at_dir, 'at_' + sub_model.split('-')[0] + '_.txt')

        if not os.path.exists(at_file_path):
            model, conv_shape, base_model = using_model(frame_num,
                                                        height,
                                                        config.ctc_class)
            base_model.load_weights(model_path)
            pred = base_model.predict(config.test_x)

            each_model_dir = os.path.join(ctc_frame_prob_dir, sub_model.split('-')[0])
            create_folder(each_model_dir)

            final_out = {}
            for id, audio in enumerate(test_id):
                prob_file = os.path.join(each_model_dir, audio + '.txt')
                np.savetxt(prob_file, pred[id], fmt='%.4f')

                pred_at = pred[id]
                shape = pred_at.shape
                pred_at = pred_at.reshape(1, shape[0], shape[1])
                out = K.get_value(K.ctc_decode(pred_at,
                                               input_length=np.ones(pred_at.shape[0]) * pred_at.shape[1], )[0][0])


                pred_result = []
                for x in out[0]:
                    if x < len(weak_label_set):
                        pred_result.append(weak_label_set[x])
                final_out[audio] = pred_result

            with open(at_file_path, 'w') as file:
                for key in final_out.keys():
                    file.write(str(key) + '\t')
                    for i in range(len(final_out[key])):
                        file.write(final_out[key][i] + '\t')
                    file.write('\r\n')
                    file.flush()
            K.clear_session()
        else:
            print('Done: ', at_file_path)

    print("Prediction finished!")



from sklearn import metrics
from nltk.translate.bleu_score import sentence_bleu
def evaluate_audio_tagging(pred_event_at_dir, only_final=False, only_final_epoch=None):

    weak_label_set = config.weak_label_set
    sequential_labels = config.train_val_test_data['testing_sequential_labels']

    sequential_events = []
    for each in sequential_labels:
        each_events = [weak_label_set[i] for i in each if i < len(weak_label_set)]
        sequential_events.append(each_events)

    test_id = config.test_id
    tagging_truth_label_matrix = config.tagging_truth_label_matrix

    for file in os.listdir(pred_event_at_dir):
        if str(only_final_epoch) in file or not only_final:
            filepath = os.path.join(pred_event_at_dir, file)
            id_list = []
            events_list = []
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    if len(line.split('\n')[0]):
                        part = line.split('\t\n')[0].split('\t')
                        audio_id = part[0]
                        events = part[1:]
                        id_list.append(audio_id)
                        if len(events):
                            events_list.append(events)
                        else:
                            events_list.append([])

            bleu_list = []
            for pred_y_only_event, each_seq_y_only_event in zip(events_list, sequential_events):
                reference = [each_seq_y_only_event]
                candidate = pred_y_only_event
                bleu = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                bleu_list.append(bleu)
            ave_bleu = sum(bleu_list) / len(bleu_list)

            pre_tagging_label_matrix = np.zeros(tagging_truth_label_matrix.shape)
            for num, each_id in enumerate(test_id):
                events = events_list[id_list.index(each_id)]
                if len(events):
                    for event in events:
                        if event in weak_label_set:
                            pre_tagging_label_matrix[num, weak_label_set.index(event)] = 1

            auc_list = []
            for k in range(tagging_truth_label_matrix.shape[-1]):
                pre_tagging_label_matrix_each_event = pre_tagging_label_matrix[:, k]
                tagging_truth_label_matrix_each_event = tagging_truth_label_matrix[:, k]

                if np.sum(tagging_truth_label_matrix_each_event):
                    test_auc = metrics.roc_auc_score(tagging_truth_label_matrix_each_event,
                                                     pre_tagging_label_matrix_each_event)
                    auc_list.append(test_auc)
            ave_auc = np.mean(auc_list)

            tp = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix > 1.5)
            fn = np.sum(tagging_truth_label_matrix - pre_tagging_label_matrix > 0.5)
            fp = np.sum(pre_tagging_label_matrix - tagging_truth_label_matrix > 0.5)
            prec = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            fvalue = 2 * (prec * recall) / (prec + recall)

            print('F-score: ', fvalue, ' AUC:', ave_auc, ' BLEU: ', ave_bleu)







