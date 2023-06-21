import numpy as np
import h5py, os, pickle, torch
import csv
import time
import framework.config as config
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool


class DataGenerator_Noiseme_2tokens(object):
    def __init__(self, start_end_labels=False, batch_size=config.batch_size, seed=9):

        # print(config.training_val_test_file)
        # D:\Yuanbo\Dataset\noiseme\using_noiseme_sequential_weak_labels_1024frames_AST_all.pickle

        with open(config.training_val_test_file, 'rb') as f:
            self.train_val_test_data = pickle.load(f)
            # print(self.train_val_test_data.keys())
        # dict_keys(['validation_weak_matrix', 'validation_audio_ids', 'validation_feature',
        # 'validation_sequential_labels', 'validation_seq_length', 'validation_weak_length',
        # 'testing_weak_matrix', 'testing_audio_ids', 'testing_feature', 'testing_sequential_labels',
        # 'testing_seq_length', 'testing_weak_length', 'training_weak_matrix', 'training_audio_ids',
        # 'training_feature', 'training_sequential_labels', 'training_seq_length', 'training_weak_length',
        # 'weak_label_set'])

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(seed)
        self.train_audio_indexes = self.train_val_test_data['training_audio_ids']

        self.x = self.train_val_test_data['training_feature']

        self.x_val = self.train_val_test_data['validation_feature']
        self.val_audio_indexes = self.train_val_test_data['validation_audio_ids']

        self.start_end_labels = start_end_labels
        if start_end_labels:
            self.y = self.train_val_test_data['train_start_end_sequence_labels']
            self.y_val = self.train_val_test_data['val_start_end_sequence_labels']
            self.all_labels_set = self.train_val_test_data['start_end_sequence_labels_set']
        else:
            self.y = self.train_val_test_data['training_sequential_labels']
            self.y_val = self.train_val_test_data['validation_sequential_labels']
            self.all_labels_set = self.train_val_test_data['weak_label_set']

        self.batch_size = batch_size

        self.v2i = {}
        self.i2v = {}

        self.start_string = '<GO>'
        self.end_string = '<EOS>'
        self.pad_string = '<PAD>'
        self.words_num = 0
        self.max_length = 0
        self.start_string_reverse = '<GO_reverse>'

    def label_set(self):
        token_list = [self.start_string, self.start_string_reverse, self.end_string]
        vocab = set([i for i in self.all_labels_set] + token_list)

        self.v2i = {v: i for i, v in enumerate(sorted(list(vocab)), start=1)}
        self.v2i[self.pad_string] = 0  # PAD_ID
        vocab.add(self.pad_string)
        self.i2v = {i: v for v, i in self.v2i.items()}
        # print('self.i2v:', self.i2v)
        # print('self.v2i:', self.v2i)

        self.words_num = len(self.i2v.keys())

        val_y = self.train_val_test_data['validation_sequential_labels']
        max_length = val_y.shape[1]
        self.max_length = max_length + 2  # add '<EOS>' and '<GO>'
        # 这里只需要加2就行，因为不管是 正序还是逆序，开始的start都只是占一个字符，同样的位置，长度是不变的


    def seq2id(self, new_y):
        sequence = [self.v2i[word] for word in new_y]
        if len(sequence) < self.max_length:
            sequence.extend([self.v2i[self.pad_string]
                             for i in range(self.max_length - len(sequence))])
        return sequence

    def train_process(self, each):

        if 'nobacknoise' in config.training_val_test_file:
            each = [j - 1 for j in each if j != 1001]

        label_text = [self.all_labels_set[k] for k in each if k < len(self.all_labels_set)]
        new_y = []
        insert_len = len(label_text) - 1
        new_y.append(self.start_string)
        if insert_len:
            for i in range(insert_len):
                new_y.append(label_text[i])
        new_y.append(label_text[-1])
        new_y.append(self.end_string)

        new_y_reverse = new_y[::-1]
        new_y_reverse_context = new_y_reverse[1:-1]
        new_y_reverse_context.insert(0, self.start_string_reverse)
        new_y_reverse_context.append(self.end_string)


        sequence = self.seq2id(new_y)
        sequence_reverse = self.seq2id(new_y_reverse)
        sequence_reverse_context = self.seq2id(new_y_reverse_context)

        return new_y, sequence, sequence_reverse, sequence_reverse_context

    def generate_train(self):
        batch_size = self.batch_size
        audio_indexes = [i for i in range(len(self.train_audio_indexes))]
        audios_num = len(self.train_audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size
            # print(batch_audio_indexes)
            # print(type(batch_audio_indexes))

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            # start = time.time()^M
            pool_size = batch_size
            pool = ThreadPool(pool_size)
            pool_output = pool.map(self.train_process, batch_y)
            pool.close()
            pool.join()

            batch_y_len = []
            batch_y_sequence = []
            batch_y_sequence_reverse = []
            batch_y_sequence_reverse_context = []
            for each in pool_output:
                new_y, sequence, sequence_reverse, sequence_reverse_context = each[0], each[1], each[2], each[3]
                batch_y_len.append(len(new_y))
                batch_y_sequence.append(sequence)
                batch_y_sequence_reverse.append(sequence_reverse)
                batch_y_sequence_reverse_context.append(sequence_reverse_context)

            yield batch_x, np.array(batch_y_sequence), np.array(batch_y_len), \
                  np.array(batch_y_sequence_reverse), np.array(batch_y_sequence_reverse_context)

    def generate_test(self, max_iteration=None):

        self.train_val_test_data = pickle.load(open(config.training_val_test_file, 'rb'))

        test_x = self.train_val_test_data['testing_feature']

        weak_label_set = self.train_val_test_data['weak_label_set']
        test_id = self.train_val_test_data['testing_audio_ids']
        test_y_seq = self.train_val_test_data['testing_sequential_labels']

        seq_y_len = []
        all_y_sequence = []
        all_y_sequence_reverse = []
        for each in test_y_seq:
            if 'nobacknoise' in config.training_val_test_file:
                text_label = [weak_label_set[j-1] for j in each if j !=1001]
            else:
                text_label = [weak_label_set[j] for j in each if j != 1001]

            new_y = []
            insert_len = len(text_label) - 1
            new_y.append(self.start_string)
            if insert_len:
                for i in range(insert_len):
                    if self.start_end_labels:
                        new_y.append(text_label[i] + '_s')
                        new_y.append(text_label[i] + '_e')
                    else:
                        new_y.append(text_label[i])
            if self.start_end_labels:
                new_y.append(text_label[-1] + '_s')
                new_y.append(text_label[-1] + '_e')
            else:
                new_y.append(text_label[-1])
            new_y.append(self.end_string)

            seq_y_len.append(len(new_y))
            sequence = [self.v2i[word] for word in new_y]
            all_y_sequence.append(sequence)

        batch_size = self.batch_size
        audio_indexes = [i for i in range(len(test_id))]
        audios_num = len(test_id)

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break
            # this will stop the loop, but the training does not have this, so the training keeps going

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = test_x[batch_audio_indexes]

            batch_y_weak = []
            batch_audio_names = []
            batch_y_seq = []
            batch_y_seq_len = []
            for k in batch_audio_indexes:
                # batch_y_weak.append(test_y_weak[k])
                batch_audio_names.append(test_id[k])
                batch_y_seq.append(all_y_sequence[k])
                batch_y_seq_len.append(seq_y_len[k])

            yield batch_x, batch_y_seq, batch_y_seq_len, batch_audio_names


    def generate_validate(self, data_type, max_iteration=None):
        if data_type == 'training':
            audio_indexes = [i for i in range(len(self.train_audio_indexes))]
            audios_num = len(self.train_audio_indexes)
            self.x_using = self.x
            self.y_using = self.y
        elif data_type == 'validation':
            audio_indexes = [i for i in range(len(self.val_audio_indexes))]
            audios_num = len(self.val_audio_indexes)
            self.x_using = self.x_val
            self.y_using = self.y_val
        else:
            raise Exception('Invalid data_type!')

        self.validate_random_state.shuffle(audio_indexes)

        batch_size = self.batch_size

        print('Number of {} audios in {}'.format(audios_num, data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break
            # this will stop the loop, but the training does not have this, so the training keeps going

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x_using[batch_audio_indexes]
            batch_y = self.y_using[batch_audio_indexes]

            if data_type == 'training':
                batch_audio_names = [self.train_audio_indexes[k] for k in batch_audio_indexes]
            elif data_type == 'validation':
                batch_audio_names = [self.val_audio_indexes[k] for k in batch_audio_indexes]

            yield batch_x, batch_y, batch_audio_names








