import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.layers import merge
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import RepeatVector
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras import regularizers
from keras import backend as K


vocab_size = 10000
embbed_size = 200
chebi_embbed_size = 100
wordnet_embbed_size = 47
LSTM_units = 200
sigmoid_units = 100
sigmoid_l2_reg = 0.00001
dropout1 = 0.5
#n_classes = 19
n_classes = 5
max_sentence_length = 80
max_ancestors_length = 10

words_channel = True
wordnet_channel = False
ancestors_channel = False

# https://github.com/philipperemy/keras-attention-mechanism
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, max_sentence_length))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(max_sentence_length, activation='softmax')(a)
    #if SINGLE_ATTENTION_VECTOR:
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def get_words_channel_single(words_input, embedding_matrix):
    e_words = embedding_matrix
    e_words = e_words(words_input)

    #attention = attention_3d_block(e_words)

    e_words = Dropout(dropout1)(e_words)
    words_lstm = Bidirectional(LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                           kernel_regularizer=regularizers.l2(sigmoid_l2_reg)))(e_words)
    words_lstm = Dropout(dropout1)(words_lstm)
    words_pool = GlobalMaxPooling1D()(words_lstm)
    return words_pool



def get_words_channel_conc(words_input, embedding_matrix):
    concatenate = keras.layers.concatenate([words_input[0], words_input[1]], axis=-1)
    e_words = embedding_matrix
    e_words = e_words(concatenate)
    #e_words = Dropout(dropout1)(e_words)

    words_lstm = Bidirectional(LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                           kernel_regularizer=regularizers.l2(sigmoid_l2_reg)))(e_words)
    words_pool = GlobalMaxPooling1D()(words_lstm)
    return words_pool

def get_words_channel(words_input, embedding_matrix):


    # e_words_left = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
    #                   trainable=False)
    # e_words_left.build((None,))
    # e_words_left.set_weights([embedding_matrix])
    e_words_left = embedding_matrix
    e_words_left = e_words_left(words_input[0])

    e_words_left = Dropout(dropout1)(e_words_left)

    # e_words_right = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
    #                    trainable=False)
    # e_words_right.build((None,))
    # e_words_right.set_weights([embedding_matrix])
    e_words_right = embedding_matrix
    e_words_right = e_words_right(words_input[1])

    e_words_right = Dropout(dropout1)(e_words_right)

    words_lstm_left = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                           kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_words_left)
    words_lstm_left = Dropout(dropout1)(words_lstm_left)

    words_lstm_right = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                            kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_words_right)
    words_lstm_right = Dropout(dropout1)(words_lstm_right)

    words_pool_left = GlobalMaxPooling1D()(words_lstm_left)
    words_pool_right = GlobalMaxPooling1D()(words_lstm_right)

    # we_hidden = Dense(sigmoid_units, activation='sigmoid',
    #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    # we_hidden = Dropout(dropout1)(we_hidden)
    return (words_pool_left, words_pool_right)


def get_chebi_channel(chebi_input, id_to_index):
    e_ancestors_left = Embedding(len(id_to_index), chebi_embbed_size, input_length=max_ancestors_length,
                                 trainable=True)
    e_ancestors_left.build((None,))
    e_ancestors_left = e_ancestors_left(chebi_input[0])

    e_ancestors_left = Dropout(0.5)(e_ancestors_left)

    e_ancestors_right = Embedding(len(id_to_index), chebi_embbed_size, input_length=max_ancestors_length,
                                  trainable=True)
    e_ancestors_right.build((None,))
    # e_right.set_weights([embedding_matrix])
    e_ancestors_right = e_ancestors_right(chebi_input[0])

    e_ancestors_right = Dropout(0.5)(e_ancestors_right)

    #ancestors_lstm_left = LSTM(LSTM_units, input_shape=(max_ancestors_length, chebi_embbed_size), return_sequences=True,
    #                           kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_ancestors_left)
    #ancestors_lstm_right = LSTM(LSTM_units, input_shape=(max_ancestors_length, chebi_embbed_size),
    #                            return_sequences=True,
    #                            kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_ancestors_right)


    #ancestors_pool_left = GlobalMaxPooling1D()(ancestors_lstm_left)
    #ancestors_pool_right = GlobalMaxPooling1D()(ancestors_lstm_right)

    ancestors_dense_left = Dense(LSTM_units, input_shape=(max_ancestors_length, chebi_embbed_size))(e_ancestors_left)
    ancestors_dense_right = Dense(LSTM_units, input_shape=(max_ancestors_length, chebi_embbed_size))(e_ancestors_right)
    return ancestors_dense_left, ancestors_dense_right

def get_model(embedding_matrix, wordnet_emb, id_to_index):
    inputs = []
    pool_layers = []
    if words_channel:
        words_input_left = Input(shape=(max_sentence_length,), name='left_words')
        words_input_right = Input(shape=(max_sentence_length,), name='right_words')

        inputs += [words_input_left, words_input_right]

        #words_input = Input(shape=(max_sentence_length,), name='words')
        #inputs += [words_input]

        words_pool_left, words_pool_right = get_words_channel((words_input_left, words_input_right),
                                                              embedding_matrix)
        pool_layers += [words_pool_left, words_pool_right]
        #words_pool = get_words_channel_conc((words_input_left, words_input_right),
        #                                                      embedding_matrix)
        #words_pool = get_words_channel_single(words_input, embedding_matrix)
        #pool_layers += [words_pool]

    if wordnet_channel:
        wordnet_left = Input(shape=(max_sentence_length,), name='left_wordnet')
        wordnet_right = Input(shape=(max_sentence_length,), name='right_wordnet')

        inputs += [wordnet_left, wordnet_right]

        e_wn_left = Embedding(len(wordnet_emb), wordnet_embbed_size, input_length=max_sentence_length,
                                 trainable=True)
        e_wn_left.build((None,))
        e_wn_left = e_wn_left(wordnet_left)

        e_wn_left = Dropout(0.5)(e_wn_left)

        e_wn_right = Embedding(len(wordnet_emb), wordnet_embbed_size, input_length=max_sentence_length,
                                  trainable=True)
        e_wn_right.build((None,))
        #e_words_right.set_weights([embedding_matrix])
        e_wn_right = e_wn_right(wordnet_right)

        e_wn_right = Dropout(0.5)(e_wn_right)

        wn_lstm_left = LSTM(wordnet_embbed_size, input_shape=(max_sentence_length, wordnet_embbed_size), return_sequences=True,
                               kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_wn_left)
        wn_lstm_right = LSTM(wordnet_embbed_size, input_shape=(max_sentence_length, wordnet_embbed_size), return_sequences=True,
                                kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_wn_right)

        wn_pool_left = GlobalMaxPooling1D()(wn_lstm_left)
        wn_pool_right = GlobalMaxPooling1D()(wn_lstm_right)

        # we_hidden = Dense(sigmoid_units, activation='sigmoid',
        #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
        # we_hidden = Dropout(dropout1)(we_hidden)
        pool_layers += [wn_pool_left, wn_pool_right]

    if ancestors_channel:
        # use just one chain
        # do not use LSTM - order is always the same
        ancestors_input_left = Input(shape=(max_ancestors_length,), name='left_ancestors')
        ancestors_input_right = Input(shape=(max_ancestors_length,), name='right_ancestors')
        ancestors_common = Input(shape=(max_ancestors_length,), name='common_ancestors')
        inputs += [ancestors_input_left, ancestors_input_right]

        ancestors_pool_left, ancestors_pool_right = get_chebi_channel((ancestors_input_left, ancestors_input_right),
                                                                      id_to_index)
        pool_layers += [ancestors_pool_left, ancestors_pool_right]

    if len(pool_layers) > 1:
        concatenate = keras.layers.concatenate(pool_layers, axis=-1)
    else:
        concatenate = pool_layers[0]

    final_hidden = Dense(sigmoid_units, activation='sigmoid',  # , kernel_initializer="random_normal",
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg), )(concatenate)
    output = Dense(n_classes, activation='softmax',
                   name='output')(final_hidden)
    model = Model(inputs=inputs, outputs=[output])
    # model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence_length))
    # model.add(Flatten())

    # model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.1),
                  #optimizer=Adam(),
                  #optimizer=RMSprop(),
                  metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model

def get_words_model(embedding_matrix):
    input = Input(shape=((max_sentence_length*2)-1,), name="input")

    emb = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                        trainable=False)
    emb.build((None,))
    emb.set_weights([embedding_matrix])
    emb = emb(input)


    lstm = LSTM(LSTM_units, input_shape=((max_sentence_length*2)-1, embbed_size), #, return_sequences=True,
                kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(emb)

    #pool_left = GlobalMaxPooling1D()(lstm_left)
    #pool_right = GlobalMaxPooling1D()(lstm_right)

    #concatenate = keras.layers.concatenate([pool_left, pool_right], axis=-1)
    # we_hidden = Dense(sigmoid_units, activation='sigmoid',
    #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    # we_hidden = Dropout(dropout1)(we_hidden)
    final_hidden = Dense(sigmoid_units, activation='sigmoid',  # , kernel_initializer="random_normal",
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg), )(lstm)
    output = Dense(n_classes, activation='softmax',
                   name='output')(final_hidden)
    model = Model(inputs=input, outputs=output)

    # model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.1),
                  #optimizer=Adam(0.001),
                  metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model


def get_xu_model(embedding_matrix):
    #input_left = Input(shape=(max_sentence_length, embbed_size),  name='left_input')
    #input_right = Input(shape=(max_sentence_length, embbed_size),  name='right_input')

    input_left = Input(shape=(max_sentence_length,), name='left_words')
    input_right = Input(shape=(max_sentence_length,), name='right_words')

    e_left = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                       trainable=False)
    e_left.build((None,))
    e_left.set_weights([embedding_matrix])
    e_left = e_left(input_left)

    e_left = Dropout(0.5)(e_left)

    e_right = Embedding(len(embedding_matrix), embbed_size, input_length=max_sentence_length,
                        trainable=False)
    e_right.build((None,))
    e_right.set_weights([embedding_matrix])
    e_right = e_right(input_right)

    e_right = Dropout(0.5)(e_right)

    lstm_left = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                     kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_left)
    lstm_right = LSTM(LSTM_units, input_shape=(max_sentence_length, embbed_size), return_sequences=True,
                      kernel_regularizer=regularizers.l2(sigmoid_l2_reg))(e_right)

    pool_left = GlobalMaxPooling1D()(lstm_left)
    pool_right = GlobalMaxPooling1D()(lstm_right)

    concatenate = keras.layers.concatenate([pool_left, pool_right], axis=-1)
    #we_hidden = Dense(sigmoid_units, activation='sigmoid',
    #                  kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    #we_hidden = Dropout(dropout1)(we_hidden)
    final_hidden = Dense(sigmoid_units, activation='sigmoid', #, kernel_initializer="random_normal",
                         kernel_regularizer=regularizers.l2(sigmoid_l2_reg),)(concatenate)
    output = Dense(n_classes, activation='softmax',
                   name='output')(final_hidden)
    model = Model(inputs=[input_left, input_right], outputs=[output])
    #model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence_length))
    #model.add(Flatten())

    #model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  #optimizer=SGD(0.1),
                  optimizer=Adam(),
                  metrics=['accuracy', precision, recall, f1])
    print(model.summary())
    return model

#https://github.com/fchollet/keras/issues/5400
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    # print(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,1:], 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[...,1:], 0, 1)))
        p = true_positives / (predicted_positives + K.epsilon())
        return p

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
        r = true_positives / (possible_positives + K.epsilon())
        return r
    precision_v = precision(y_true, y_pred)
    recall_v = recall(y_true, y_pred)
    return (2.0*precision_v*recall_v)/(precision_v+recall_v)

