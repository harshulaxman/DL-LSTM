# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying entities such as person names, organizations, locations, dates, and other predefined categories from unstructured text data. Traditional rule-based and statistical approaches often struggle with handling contextual dependencies and sequential information in sentences.

The objective of this project is to develop a Deep Learning-based Named Entity Recognition model using Long Short-Term Memory (LSTM) networks. The model should be capable of learning contextual relationships within sequences of words and accurately predicting entity labels for each token in a sentence.

The system will take annotated text data as input, preprocess it into suitable numerical representations (such as word embeddings), and train an LSTM model to perform sequence labeling. The performance of the model will be evaluated based on metrics such as accuracy, precision, recall, and F1-score.

The developed model aims to improve entity recognition performance by leveraging the sequential learning capability of LSTMs and handling long-range dependencies in text.

<img width="1136" height="670" alt="image" src="https://github.com/user-attachments/assets/3ac2fa1c-7c3b-4234-83d8-befdfd9283fd" />


## DESIGN STEPS
### STEP 1: 

Load data, create word/tag mappings, and group sentences.

### STEP 2: 

Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3: 

Define dataset and DataLoader for batching.

### STEP 4: 

Build a bidirectional LSTM model for sequence tagging.

### STEP 5: 

Train the model over multiple epochs, tracking loss.


## Program
### Name: Athul Krishna A V

### Register Number: 212225240017
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ner_dataset.csv", encoding="latin1")
df["Sentence #"] = df["Sentence #"].ffill()

words = list(set(df["Word"].values))
tags = list(set(df["Tag"].values))

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["<PAD>"] = 0

class SentenceGetter:
    def __init__(self, data):
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = data.groupby("Sentence #", sort=False).apply(agg_func)
        self.sentences = [s for s in self.grouped]

getter = SentenceGetter(df)
sentences = getter.sentences

class NERDataset(Dataset):
    def __init__(self, sentences, word2idx, tag2idx, max_len=50):
        self.sentences = sentences
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        word_ids = [self.word2idx.get(w[0], self.word2idx["<UNK>"]) for w in sentence]
        tag_ids = [self.tag2idx[w[1]] for w in sentence]

        word_ids = word_ids[:self.max_len]
        tag_ids = tag_ids[:self.max_len]

        pad_len = self.max_len - len(word_ids)
        word_ids = word_ids + [self.word2idx["<PAD>"]] * pad_len
        tag_ids = tag_ids + [self.tag2idx["<PAD>"]] * pad_len

        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)

train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

train_dataset = NERDataset(train_sentences, word2idx, tag2idx)
test_dataset = NERDataset(test_sentences, word2idx, tag2idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=128, dropout_rate=0.3):
        super(BiLSTM_NER, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # bidirectional=True doubles hidden dimension (hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)           # (batch_size, seq_len, hidden_dim * 2)
        logits = self.fc(self.dropout(lstm_out))     # (batch_size, seq_len, num_tags)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTM_NER(
    vocab_size=len(word2idx),
    num_tags=len(tag2idx),
    embedding_dim=128,
    hidden_dim=128
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epoch_losses = []
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for words_batch, tags_batch in train_loader:
        words_batch, tags_batch = words_batch.to(device), tags_batch.to(device)

        optimizer.zero_grad()
        logits = model(words_batch)
        
        loss = criterion(logits.view(-1, logits.shape[-1]), tags_batch.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss) # Track the loss here
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# Plotting the Epoch vs. Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.show()

idx2tag = {v: k for k, v in tag2idx.items()}

def compare_ner_predictions(model, labeled_sentence, word2idx, idx2tag, max_len=50):
    model.eval()
    
    sentence_words = [item[0] for item in labeled_sentence]
    true_tags = [item[1] for item in labeled_sentence]
    
    word_ids = [word2idx.get(w, word2idx["<UNK>"]) for w in sentence_words]
    
    pad_len = max_len - len(word_ids)
    padded_word_ids = word_ids + [word2idx["<PAD>"]] * pad_len
    
    input_tensor = torch.tensor([padded_word_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.argmax(logits, dim=-1)
        
    pred_tags = [idx2tag[idx.item()] for idx in preds[0][:len(sentence_words)]]
    
    return zip(sentence_words, true_tags, pred_tags)

sample_data = [
    ("The", "O"), 
    ("Prime", "B-per"), 
    ("Minister", "I-per"), 
    ("of", "O"), 
    ("India", "B-geo"), 
    ("visited", "O"), 
    ("New", "B-geo"), 
    ("Delhi", "I-geo"), 
    ("on", "O"), 
    ("Monday", "B-tim"), 
    (".", "O")
]

comparisons = compare_ner_predictions(model, sample_data, word2idx, idx2tag)
print("Name: Harsshitha lakshmanan\nReg No: 212223230075")
print("-"*48)
print(f"\n{'WORD':<15} | {'TRUE TAG':<12} | {'PREDICTED TAG'}")
print("-" * 48)
for word, true_tag, pred_tag in comparisons:
    match = " " if true_tag == pred_tag else "*"
    print(f"{word:<15} | {true_tag:<12} | {pred_tag} {match}")
```
### OUTPUT

## Loss Vs Epoch Plot

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/ab946ef4-3c69-4022-9c30-b1d4de6aaf4f" />

### Sample Text Prediction


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
