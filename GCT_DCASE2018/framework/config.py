import os, pickle
import numpy as np

dataset_dir = os.getcwd()
training_val_file = os.path.join(dataset_dir, 'Dataset', 'train_val_sequential_weak_labels_feature.pickle')
test_file = os.path.join(dataset_dir, 'Dataset', 'test_sequential_weak_labels_feature.pickle')

batch_size = 64
mel_bins = 768  # 128
cuda = 1
show_training = 1

only_save_best = False


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

# d_model = 128  # Embedding Size
# d_ff = 512  # FeedForward dimension
# d_k = d_v = 64  # dimension of K(=Q), V
# n_heads = 4  # number of heads in Multi-Head Attention

endswith = '.pth'

test_data = pickle.load(open(test_file, 'rb'))
weak_label_set = test_data['weak_label_set']
test_id = test_data['audio_names']
tagging_truth_label_matrix = []
for each in test_data['weak_labels']:
    each_label = np.zeros((len(weak_label_set)))
    for event in each:
        each_label[event] = 1
    tagging_truth_label_matrix.append(each_label)
tagging_truth_label_matrix = np.array(tagging_truth_label_matrix)



