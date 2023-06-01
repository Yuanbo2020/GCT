# coding= utf-8
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from tensorflow.keras.layers import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from tensorflow.keras.layers import Input, Multiply
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import *



class Yuanbo_ModelCheckpoint(Callback):
    def __init__(self, filepath, verbose=0,  save_weights_only=False,
                 last_epochs=1000,  period=1):
        super(Yuanbo_ModelCheckpoint, self).__init__()
        self.verbose = verbose
        self.filepath = filepath

        self.last_epochs = last_epochs

        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # print "logs:", logs   # logs: {'loss': '395.92644727183625', 'val_loss': '279.26708211778083'}
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.verbose > 0:
                print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)

        if epoch > self.last_epochs:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.model.save(filepath, overwrite=True)



def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out


def slice1(x):
    return x[:, :, :, :64]


def slice2(x):
    return x[:, :, :, 64:]


def slice1_output_shape(input_shape):
    return tuple([input_shape[0], input_shape[1], input_shape[2], 64])


def slice2_output_shape(input_shape):
    return tuple([input_shape[0], input_shape[1], input_shape[2], 64])


#################################################### 128 ##########################################
def block_128(input):
    cnn = Conv2D(256, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1_128, output_shape=slice1_output_shape_128)(cnn)
    cnn2 = Lambda(slice2_128, output_shape=slice2_output_shape_128)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out


def slice1_128(x):
    return x[:, :, :, :128]


def slice2_128(x):
    return x[:, :, :, 128:]


def slice1_output_shape_128(input_shape):
    return tuple([input_shape[0], input_shape[1], input_shape[2], 128])


def slice2_output_shape_128(input_shape):
    return tuple([input_shape[0], input_shape[1], input_shape[2], 128])
#########################################################################################################


def ctc_lambda_func(args):
    from keras import backend as K
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def model_icassp2019(width, height, ctc_class, truth_event_len=None):
    input_logmel = Input(shape=(width, height), name='in_layer')  # (N, 240, 64)
    a1 = Reshape((width, height, 1))(input_logmel)  # (N, 240, 64, 1)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 4), name='block1_out')(a1)  # (N, 240, 32, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 4), name='block2_out')(a1)  # (N, 240, 16, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 4), name='block3_out')(a1)  # (N, 240, 8, 128)

    global conv_shape
    conv_shape = a1.get_shape()
    a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(a1)

    rnn_size = 64

    rnnout = Bidirectional(GRU(rnn_size, return_sequences=True))(a1)
    rnnout = Activation(activation='linear')(rnnout)
    rnnout_gate = Bidirectional(GRU(rnn_size, return_sequences=True))(a1)
    rnnout_gate = Activation(activation='sigmoid')(rnnout_gate)
    a2 = Multiply()([rnnout, rnnout_gate])

    x = Dense(ctc_class, kernel_initializer='he_normal', activation='softmax')(a2)
    base_model = Model(inputs=input_logmel, outputs=x)

    n_len = truth_event_len

    labels = Input(name='the_labels', shape=[n_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_logmel, labels, input_length, label_length],
                  outputs=[loss_out])
    model.summary()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
    return model, conv_shape, base_model

