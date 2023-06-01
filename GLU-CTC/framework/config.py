import os, pickle


dataset_dir = os.getcwd()
training_val_test_file = os.path.join(dataset_dir, 'Dataset', 'Noiseme_sequential_weak_labels.pickle')

mel_bins = 64
cuda = 1
batch_size = 64

with open(training_val_test_file, 'rb') as f:
    train_val_test_data = pickle.load(f)

train_x = train_val_test_data['normal_train_x']
train_y = train_val_test_data['training_sequential_labels']

validation_x = train_val_test_data['normal_test_x']
validation_y = train_val_test_data['testing_sequential_labels']
val_truth_label_len = train_val_test_data['testing_seq_length']
train_truth_label_len = train_val_test_data['training_seq_length']


ctc_class = len(train_val_test_data['weak_label_set']) + 1

######################### testing #############################################
test_x = train_val_test_data['normal_test_x']
weak_label_set = list(train_val_test_data['weak_label_set'].values())
test_id = train_val_test_data['testing_audio_ids']
tagging_truth_label_matrix = train_val_test_data['testing_weak_matrix']





