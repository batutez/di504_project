"""
Section 13 — GRU model architecture, vocabulary building, and Dataset.

GRU Architecture (from the paper)
    Embedding(vocab_size, embedding_dim)
    -> GRU(embedding_dim, hidden_dim, batch_first=True)
    -> Dropout(0.3, fixed — not tuned)
    -> Linear(hidden_dim, num_classes=7)

Forward pass takes ONLY the last hidden state of the GRU (not all timestep
outputs), and returns raw logits — no softmax. Softmax is applied implicitly
by CrossEntropyLoss during training, and explicitly (if needed) only at
evaluation time, for probability-based metrics like AUROC.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter

def build_vocab(train_texts, vocab_size=10000):
    """
    Build a word to integer index vocabulary from the training data

    Returns
    -------
    dict[str, int]
    """
    word_counts = Counter()
    for text in train_texts:
        tokens = text.split()
        word_counts.update(tokens)
    # keeping the most frequent words, but leaving 2 slots for the special tokens reserved below.
    most_common_words = [word for word, count in word_counts.most_common(vocab_size - 2)]
    # Build the vocab dict: <PAD>=0, <UNK>=1, then every other word starting at 2.
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for index, word in enumerate(most_common_words):
        vocab[word] = index+ 2
    
    return vocab

def text_to_sequence(text, vocab, max_len=200):
    """
    Convert one cleaned text string into a fixed-length integer sequence.
    If unknown word exists, change the word with <UNK>
    If length of sequence is longer than max_len, the first max_len word is used
    If length of sequence is shorter than max_len, the sequence is added with <PAD> to quarantee fixed sequence size

    Returns
    -------
    list[int] of length max_len

    Verification:
        len(result) == max_len always, even for empty or very long input.
    """
    tokens = text.split()
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))

    return sequence
    


class TextDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-tokenized integer sequences and labels.

    Since every sequence is already fixed-length (thanks to text_to_sequence), no
    custom collate_fn will be needed
    """

    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
       return self.sequences[index], self.labels[index]



class GRUClassifier(nn.Module):
    """
    GRU-based classifier. See module docstring for exact architecture.
    Architecture is 
    1 - Embedding Layer
    2 - GRU Layer
    3- Dropout Layer (n = 0.3)
    4 - FC Layer

    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=7, dropout=0.3):
        super().__init__()

        # <PAD> lies at index 0 , so padding_idx should be equal to 0 for the embeding layer
        self.embedding = nn.Embedding(vocab_size,embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = True)
        self.dropout = nn.Dropout(0.3) # hardcoded as it is in the paper
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        _, hidden = self.gru(embedded)  # (1, batch, hid_dim)
        hidden = hidden.squeeze(0)     # (batch,hid_dim)
        dropped = self.dropout(hidden)
        logits  = self.fc(dropped)
        return logits