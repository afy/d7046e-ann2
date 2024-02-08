import re
import numpy as np
from collections import defaultdict
np.random.seed(42)

def generate_toy_dataset(num_sequences=100):
    """
    Generates a number of sequences as our dataset.
    
    Args:
     `num_sequences`: the number of sequences to be generated.
     
    Returns a list of sequences.
    """
    samples = []
    
    for _ in range(num_sequences): 
        num_tokens = np.random.randint(1, 10)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
        
    return samples


def split_to_chars(chunks):
    delimiters = " ", "\n", ":", ","
    regex_pattern = '|'.join(map(re.escape, delimiters))
    char_sequence = []
    for chunk in chunks:
        characters = []
        for word in re.split(regex_pattern, chunk):
            chars = [*word]
            for char in chars:
                characters.append(char.lower())
        characters.append('EOS')
        char_sequence.append(characters)
    return char_sequence

def sequences_to_dicts(sequences):
    """
    Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
    """
    # A bit of Python-magic to flatten a nested list
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    # Flatten the dataset
    all_words = flatten(sequences)
    
    # Count number of word occurences
    word_count = defaultdict(int)
    for word in flatten(sequences):
        word_count[word] += 1

    # Sort by frequency
    word_count = sorted(list(word_count.items()), key=lambda l: -l[1])

    # Create a list of all unique words
    unique_words = [item[0] for item in word_count]
    
    # Add UNK token to list of words
    unique_words.append('UNK')

    # Count number of sequences and number of unique words
    num_sentences, vocab_size = len(sequences), len(unique_words)

    # Create dictionaries so that we can go from word to index and back
    # If a word is not in our vocabulary, we assign it to token 'UNK'
    word_to_idx = defaultdict(lambda: num_words)
    idx_to_word = defaultdict(lambda: 'UNK')

    # Fill dictionaries
    for idx, word in enumerate(unique_words):
        # YOUR CODE HERE!
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sentences, vocab_size

from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y

    
def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []
        
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set

def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.
    
    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary
    
    Returns a 1-D numpy array of length `vocab_size`.
    """
    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)
    
    # Set the appropriate element to one
    one_hot[idx] = 1.0

    return one_hot


def one_hot_encode_sequence(sequence, vocab_size, word_to_idx):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.
    
    Args:
     `sentence`: a list of words to encode
     `vocab_size`: the size of the vocabulary
     
    Returns a 3-D numpy array of shape (num words, vocab size, 1).
    """
    # Encode each word in the sentence
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    # Reshape encoding s.t. it has shape (num words, vocab size, 1)
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    
    return encoding



def set_up_sequences(file_name = 'input.txt'):
    if file_name == 'toy':
        sequences = generate_toy_dataset()
    else:
        with open(file_name, 'r', encoding='utf-8') as file:
            doc = file.read()
        chunks = doc.split('.')

        sequences = split_to_chars(chunks)
        #print('A single sample from the Shakespeare dataset:')
        #print(shakespeare_sequences[0])

        # Whole dataset is too big to effectively train on, so let's start by grabbing the first 100 sequences
        #sequences = sequences[0:100]

    word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)

    #print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
    #print('The index of \'b\' is', word_to_idx['b'])
    #print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')
    
    return sequences, word_to_idx, idx_to_word, num_sequences, vocab_size


def set_up_datasets(sequences):
    training_set, validation_set, test_set = create_datasets(sequences, Dataset)

    print(f'We have {len(training_set)} samples in the training set.')
    print(f'We have {len(validation_set)} samples in the validation set.')
    print(f'We have {len(test_set)} samples in the test set.')

    return training_set, validation_set, test_set


def init_orthogonal(param):
    """
    Initializes weight parameters orthogonally.
    
    Refer to this paper for an explanation of this initialization:
    https://arxiv.org/abs/1312.6120
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")

    rows, cols = param.shape
    
    new_param = np.random.randn(rows, cols)
    
    if rows < cols:
        new_param = new_param.T
    
    # Compute QR factorization
    q, r = np.linalg.qr(new_param)
    
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T
    
    new_param = q
    
    return new_param

# Another support function which normalises the gradients to prevent exploding gradients
def clip_gradient_norm(gradients, max_norm=0.25):
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """ 
    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0
    
    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in gradients:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm
    
    total_norm = np.sqrt(total_norm)
    
    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for grad in gradients:
            grad *= clip_coef
    return gradients

if __name__ == "__main__":   

    with open('input.txt', 'r', encoding='utf-8') as file:
        shakespeare_doc = file.read()
    sakespeare_chunks = shakespeare_doc.split('.')

    shakespeare_sequences = split_to_chars(sakespeare_chunks)


    print('A single sample from the Shakespeare dataset:')

    print(shakespeare_sequences[0])

    # Whole dataset is too big to effectively train on, so let's start by grabbing the first 100 sequences
    sequences = shakespeare_sequences[0:100]

    word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)

    print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
    print('The index of \'b\' is', word_to_idx['b'])
    print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

    training_set, validation_set, test_set = create_datasets(sequences, Dataset)

    print(f'We have {len(training_set)} samples in the training set.')
    print(f'We have {len(validation_set)} samples in the validation set.')
    print(f'We have {len(test_set)} samples in the test set.')

    test_word = one_hot_encode(word_to_idx['a'], vocab_size)
    print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')

    test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size)
    print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')