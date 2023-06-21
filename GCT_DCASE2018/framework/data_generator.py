import numpy as np
import h5py, os, pickle, torch
import csv
import time
import framework.config as config


class DataGenerator_D2018T4(object):
    def __init__(self, start_end_labels=False, batch_size=config.batch_size, seed=1234, normalized_data=True):

        # print(batch_size)

        with open(config.training_val_file, 'rb') as f:
            self.train_val_data = pickle.load(f)

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(seed)
        self.train_audio_indexes = self.train_val_data['train_names']

        self.normalized_data=normalized_data
        if normalized_data:
            self.x = self.train_val_data['normal_train_x']
            self.x_val = self.train_val_data['normal_val_x']
        else:
            self.x = self.train_val_data['raw_train_x']
            self.x_val = self.train_val_data['raw_val_x']

        self.val_audio_indexes = self.train_val_data['val_names']

        self.start_end_labels = start_end_labels
        if start_end_labels:
            self.y = self.train_val_data['train_start_end_sequence_labels']
            self.y_val = self.train_val_data['val_start_end_sequence_labels']
            self.all_labels_set = self.train_val_data['start_end_sequence_labels_set']
        else:
            self.y = self.train_val_data['train_start_sequence_labels']
            self.y_val = self.train_val_data['val_start_sequence_labels']
            self.all_labels_set = self.train_val_data['weak_label_set']

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

        self.words_num = len(self.i2v.keys())

        if self.start_end_labels:
            max_length = max(max(self.train_val_data['train_truth_label_len_start_end']),
                                         max(self.train_val_data['val_truth_label_len_start_end']))
        else:
            val_y = self.train_val_data['val_start_sequence_labels']
            assert val_y.shape[1] == max(max(self.train_val_data['train_truth_label_len']),
                                         max(self.train_val_data['val_truth_label_len']))
            max_length = val_y.shape[1]

        self.max_length = max_length + 2  # add '<EOS>' and '<GO>'
        # 这里只需要加2就行，因为不管是 正序还是逆序，开始的start都只是占一个字符，同样的位置，长度是不变的

    def seq2id(self, new_y):
        sequence = [self.v2i[word] for word in new_y]
        if len(sequence) < self.max_length:
            sequence.extend([self.v2i[self.pad_string]
                             for i in range(self.max_length - len(sequence))])
        return sequence

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

            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size
            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            batch_y_len = []
            batch_y_sequence = []
            batch_y_sequence_reverse = []
            batch_y_sequence_reverse_context = []
            for each in batch_y:
                label_text = [self.all_labels_set[k] for k in each
                              if k < len(self.all_labels_set)]
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
                ############################################################################################

                batch_y_len.append(len(new_y))

                sequence = self.seq2id(new_y)
                batch_y_sequence.append(sequence)

                sequence_reverse = self.seq2id(new_y_reverse)
                batch_y_sequence_reverse.append(sequence_reverse)

                sequence_reverse_context = self.seq2id(new_y_reverse_context)
                batch_y_sequence_reverse_context.append(sequence_reverse_context)



            yield batch_x, np.array(batch_y_sequence), np.array(batch_y_len), \
                  np.array(batch_y_sequence_reverse), \
                  np.array(batch_y_sequence_reverse_context)


    def generate_train_for_inference_demo(self, max_iteration=None):
        batch_size = self.batch_size
        audio_indexes = [i for i in range(len(self.train_audio_indexes))]
        audios_num = len(self.train_audio_indexes)

        test_x = self.x
        weak_label_set = self.all_labels_set
        test_id = audio_indexes

        test_y_weak = self.train_val_data['train_weak_labels']

        test_y_seq = self.y

        seq_y_len = []
        all_y_sequence = []
        for each in test_y_seq:
            text_label = [self.all_labels_set[k] for k in each if k < len(self.all_labels_set)]
            new_y = []
            insert_len = len(text_label) - 1
            new_y.append(self.start_string)
            if insert_len:
                for i in range(insert_len):
                    new_y.append(text_label[i])
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
                batch_y_weak.append(test_y_weak[k])
                batch_audio_names.append(self.train_audio_indexes[k])
                batch_y_seq.append(all_y_sequence[k])
                batch_y_seq_len.append(seq_y_len[k])

            # print(batch_x, batch_y_seq, batch_y_seq_len, batch_y_weak, batch_audio_names)
            yield batch_x, batch_y_seq, batch_y_seq_len, batch_y_weak, batch_audio_names


    def generate_test(self, max_iteration=None):
        test_data = pickle.load(open(config.test_file, 'rb'))

        if self.normalized_data:
            test_x = test_data['normal_x']
        else:
            test_x = test_data['raw_data_x']

        weak_label_set = test_data['weak_label_set']
        test_id = test_data['audio_names']
        test_y_weak = test_data['weak_labels']
        test_y_seq = test_data['start_sequence_labels']

        seq_y_len = []
        all_y_sequence = []
        all_y_sequence_reverse = []
        for each in test_y_seq:
            text_label = [weak_label_set[j] for j in each]

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
                batch_y_weak.append(test_y_weak[k])
                batch_audio_names.append(test_id[k])
                batch_y_seq.append(all_y_sequence[k])
                batch_y_seq_len.append(seq_y_len[k])

            yield batch_x, batch_y_seq, batch_y_seq_len, batch_y_weak, batch_audio_names


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





