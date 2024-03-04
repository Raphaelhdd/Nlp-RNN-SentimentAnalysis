import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib
import matplotlib.pyplot as plt
# ------------------------------------------- Constants ----------------------------------------
matplotlib.use('TkAgg')
SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict

# OK
def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v_vec = np.zeros(embedding_dim,dtype=np.float64)
    counter = 0

    for word in sent.text:
        if word in word_to_vec:
            w2v_vec += word_to_vec[word]
            counter += 1
    return w2v_vec / counter if counter != 0 else w2v_vec


# OK
def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size, dtype=np.float64)
    one_hot[ind] = 1
    return one_hot


# OK
def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    all_hot = []
    for word in sent.text:
        if word in word_to_ind:
            one_hot_word = get_one_hot(size=len(word_to_ind), ind=word_to_ind[word])
            all_hot.append(one_hot_word)
    return np.sum(all_hot, axis=0) / len(all_hot)


# OK
def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word2ind = dict()
    ind = 0
    for word in words_list:
        if word not in word2ind.keys():
            word2ind[word] = ind
            ind += 1
    return word2ind


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    result = np.zeros((seq_len, embedding_dim))
    for ind, word in enumerate(sent.text):
        if ind > 51:
            break
        if word in word_to_vec:
            result[ind,:] = word_to_vec[word]
    return result

class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, 1)
        return

    def forward(self, text):
        lstm_output, (h, c) = self.lstm(text)
        linear_input = torch.cat((h[-2, :, :], h[-1, :, :]), 1)
        return self.linear(linear_input).squeeze()

    def predict(self, text):
        return torch.sigmoid(self.forward(text))


# OK
class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    # NN module is the base class of all NN in pytorch
    # embeding dim is the size of the input and 1 will be the output
    # We take the input embedding that is a one hot embedding to a single output value
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.ll = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.ll(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


# ------------------------- training functions -------------

# ok
def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    rounded_preds = torch.round(preds)
    correct = torch.sum(rounded_preds == y)
    acc = correct / len(y)
    return acc


# ok a tester
def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()  # modele en mode entrainement
    total_loss = 0
    accuracy = 0
    for inputs, labels in data_iterator:
        outputs = model(inputs.float())
        outputs = outputs.view_as(labels)# propagation avant -> forward
        loss = criterion(outputs, labels)  # ici on calcule le loss entre prediction et labels
        optimizer.zero_grad()  # met a 0 les gradients
        loss.backward()  # on revient en arriere -> backward
        optimizer.step()  # total_loss_check met a jour les gradiants calculer
    with torch.no_grad():
        for inputs, labels in data_iterator:
            outputs = model(inputs.float())
            outputs = outputs.view_as(labels)# propagation avant -> forward
            loss = criterion(outputs, labels)  # ici on calcule le loss entre prediction et labels
            total_loss += loss.item()
            accuracy += binary_accuracy(model.predict(inputs.float()).flatten(), labels)
        epoch_loss = total_loss / len(data_iterator)
        epoch_accuracy = accuracy / len(data_iterator)
    return epoch_loss, epoch_accuracy

#OK
def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()  # modele en mode eval
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in data_iterator:
            outputs = model(inputs.float())
            loss = criterion(outputs.flatten(), labels)
            total_loss += loss.item()
            accuracy += binary_accuracy(model.predict(inputs.float()).flatten(), labels)
        epoch_loss = total_loss / len(data_iterator)
        epoch_accuracy = accuracy / len(data_iterator)
    return epoch_loss, epoch_accuracy


#OK
def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = []
    for inputs, _ in data_iter:
        predictions.append(model.predict(inputs.float()).flatten())
    return torch.cat(predictions, dim=0)

#OK
def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    train_loss_lst, train_accuracy_lst, valid_loss_lst, valid_accuracy_lst = [], [], [], []
    for epoch in range(n_epochs):
        epoch_loss, epoch_accuracy = train_epoch(model, data_manager.get_torch_iterator(), optimizer, criterion)
        train_loss_lst.append(epoch_loss)
        train_accuracy_lst.append(epoch_accuracy)
        val_loss, val_accuracy = evaluate(model,data_manager.get_torch_iterator(VAL),criterion)
        valid_loss_lst.append(val_loss)
        valid_accuracy_lst.append(val_accuracy)
        print(f"Train Loss {epoch_loss} - Train valid {val_loss}")
        save_model(model,fr"C:\Users\raphh\NLP_2024\Ex3\result\{epoch}.pt",epoch,optimizer)
    return train_loss_lst, train_accuracy_lst, valid_loss_lst, valid_accuracy_lst

def plot_curves(train_values, val_values, title, ylabel,name):
    plt.plot(train_values, label='Train')
    plt.plot(val_values, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(fr"C:\Users\raphh\NLP_2024\Ex3\result/{name}.png")

def get_negatif_accuracy(model,data_manager):
    all_sentences = data_manager.sentences[TEST]
    index_negatif = data_loader.get_negated_polarity_examples(all_sentences)
    all_labels = data_manager.get_labels(TEST)
    label_negatif = [all_labels[i] for i in index_negatif]
    all_predictions = get_predictions_for_data(model,data_manager.get_torch_iterator(TEST))
    neg_predict = [all_predictions[i] for i in index_negatif]
    accuracy = binary_accuracy(torch.tensor(neg_predict),torch.tensor(label_negatif))
    return accuracy


def get_rare_words_accuracy(model,data_manager):
    all_sentences = data_manager.sentences[TEST]
    index_negatif = data_loader.get_rare_words_examples(all_sentences, data_manager.sentiment_dataset)
    all_labels = data_manager.get_labels(TEST)
    label_negatif = [all_labels[i] for i in index_negatif]
    all_predictions = get_predictions_for_data(model,data_manager.get_torch_iterator(TEST))
    neg_predict = [all_predictions[i] for i in index_negatif]
    accuracy = binary_accuracy(torch.tensor(neg_predict),torch.tensor(label_negatif))
    return accuracy

#OK
def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    lr = 0.01
    n_epoch = 20
    batches = 64
    weight_decay = 0.001
    data_manager = DataManager(batch_size=batches)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_loss_lst, train_accuracy_lst, valid_loss_lst, valid_accuracy_lst = \
        train_model(model,data_manager,n_epoch,lr,weight_decay)
    plot_curves(train_loss_lst, valid_loss_lst, title='Log-Linear One-hot Train and Validation Loss',
                ylabel='Loss',name='log_linear_one_hot_loss')
    plot_curves(train_accuracy_lst, valid_accuracy_lst, title='Log-Linear One-hot Train and Validation Accuracy',
                ylabel='Accuracy',name='log_linear_one_hot_accur')

    test_loss, test_accuracy = evaluate(model, data_manager.get_torch_iterator(TEST),nn.BCEWithLogitsLoss())
    print(f'Log-Linear One-hot:\nTest Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}')
    accuracy_negated_subset = get_negatif_accuracy(model,data_manager)
    accuracy_rare_word_subset = get_rare_words_accuracy(model,data_manager)
    print(f"Log-Linear One-hot:\nAccuracy Negated Polarity Subset - {accuracy_negated_subset:6f}\n"
          f"Accuracy Rare Words Subset - {accuracy_rare_word_subset:6f}")


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    lr = 0.01
    n_epoch = 20
    batches = 64
    weight_decay = 0.001
    data_manager = DataManager(batch_size=batches,data_type=W2V_AVERAGE,embedding_dim=300)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_loss_lst, train_accuracy_lst, valid_loss_lst, valid_accuracy_lst = \
        train_model(model,data_manager,n_epoch,lr,weight_decay)
    plot_curves(train_loss_lst, valid_loss_lst, title='Log-Linear W2V Train and Validation Loss',
                ylabel='Loss',name='log_linear_w2v_loss')
    plot_curves(train_accuracy_lst, valid_accuracy_lst, title='Log-Linear W2V Train and Validation Accuracy',
                ylabel='Accuracy',name='log_linear_w2v_loss')

    test_loss, test_accuracy = evaluate(model, data_manager.get_torch_iterator(TEST),nn.BCEWithLogitsLoss())
    print(f'Log-Linear W2V :\nTest Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}')

    accuracy_negated_subset = get_negatif_accuracy(model,data_manager)
    accuracy_rare_word_subset = get_rare_words_accuracy(model,data_manager)
    print(f"Log-Linear W2V :\nAccuracy Negated Polarity Subset - {accuracy_negated_subset:6f}\n"
          f"Accuracy Rare Words Subset - {accuracy_rare_word_subset:6f}")

def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    lr = 0.001
    n_epoch = 4
    batches = 64
    weight_decay = 0.0001
    dropout = 0.5
    data_manager = DataManager(batch_size=batches,data_type=W2V_SEQUENCE,embedding_dim=300)
    model = LSTM(embedding_dim=300,hidden_dim=100,n_layers=2,dropout=dropout)
    train_loss_lst, train_accuracy_lst, valid_loss_lst, valid_accuracy_lst = train_model(model,
                                                                             data_manager,n_epoch,lr,weight_decay)
    plot_curves(train_loss_lst, valid_loss_lst, title='LSTM W2V Train and Validation Loss',
                ylabel='Loss',name='lstm_w2v_loss')
    plot_curves(train_accuracy_lst, valid_accuracy_lst, title='LSTM W2V Train and Validation Accuracy',
                ylabel='Accuracy',name='lstm_w2v_loss')

    test_loss, test_accuracy = evaluate(model, data_manager.get_torch_iterator(TEST), nn.BCEWithLogitsLoss())
    print(f'LSTM W2V :\nTest Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}')

    accuracy_negated_subset = get_negatif_accuracy(model,data_manager)
    accuracy_rare_word_subset = get_rare_words_accuracy(model,data_manager)
    print(f"LSTM W2V :\nAccuracy Negated Polarity Subset - {accuracy_negated_subset:6f}\n"
          f"Accuracy Rare Words Subset - {accuracy_rare_word_subset:6f}")

if __name__ == '__main__':
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
