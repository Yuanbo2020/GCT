import os, pickle
import numpy as np

dataset_dir = os.getcwd()
training_val_test_file = os.path.join(dataset_dir, 'Dataset', 'noiseme_sequential_weak_labels_nobacknoise_1024frames.pickle')


batch_size = 64
mel_bins = 128
cuda = 1
show_training = 1

# transformer
dropout = 0.2

d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

# d_model = 256  # Embedding Size
# d_ff = 1024  # FeedForward dimension
# d_k = d_v = 64  # dimension of K(=Q), V
# n_heads = 4  # number of heads in Multi-Head Attention

# d_model = 1  # Embedding Size
# d_ff = 1  # FeedForward dimension
# d_k = d_v = 1  # dimension of K(=Q), V
# n_heads = 1  # number of heads in Multi-Head Attention

endswith = '.pth'


test_data = pickle.load(open(training_val_test_file, 'rb'))
weak_label_set = test_data['weak_label_set']
test_id = test_data['testing_audio_ids']
# tagging_truth_label_matrix = []
# for each in test_data['weak_labels']:
#     each_label = np.zeros((len(weak_label_set)))
#     for event in each:
#         each_label[event] = 1
#     tagging_truth_label_matrix.append(each_label)
tagging_truth_label_matrix = test_data['testing_weak_matrix']
# print(tagging_truth_label_matrix.shape)  # (505, 33)

