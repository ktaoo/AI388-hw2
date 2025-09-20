# models.py

from collections import defaultdict
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from nltk.metrics import edit_distance


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]

class TypoCorrector:
    def __init__(self, word_indexer, max_edit_distance=2):
        self.word_indexer = word_indexer
        self.max_edit_distance = max_edit_distance
        self.correction_cache = {}
        
        # Build prefix index based on the first 3 characters
        self.prefix_to_words = defaultdict(list)

        all_words = []
        for i in range(len(word_indexer)):
            word = word_indexer.get_object(i)
            if word not in ["PAD", "UNK"]:
                all_words.append(word)

        for word in all_words:
            if len(word) >= 3:
                prefix = word[:3].lower()
                self.prefix_to_words[prefix].append(word)
    
    def find_correction(self, typo_word):
        if typo_word in self.correction_cache:
            return self.correction_cache[typo_word]
        
        if len(typo_word) < 3:
            self.correction_cache[typo_word] = None
            return None
        
        prefix = typo_word[:3].lower()
        candidates = self.prefix_to_words.get(prefix, [])
        if not candidates:
            self.correction_cache[typo_word] = None
            return None
        
        best_word = None
        best_distance = float('inf')
        
        for candidate in candidates:
            if abs(len(candidate) - len(typo_word)) > self.max_edit_distance:
                continue
                
            distance = edit_distance(typo_word.lower(), candidate.lower())
            if distance <= self.max_edit_distance and distance < best_distance:
                best_distance = distance
                best_word = candidate
        
        self.correction_cache[typo_word] = best_word
        return best_word

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, hidden_size: int = 64, num_classes: int = 2):
        super(DeepAveragingNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=False)
        self.embedding_dim = word_embeddings.get_embedding_length()
        self.word_indexer = word_embeddings.word_indexer
        
        self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.to(self.device)
        
    def forward(self, word_indices):
        embedded = self.embedding(word_indices)
        
        mask = (word_indices != self.word_indexer.index_of("PAD")).float().unsqueeze(-1)
        masked_embedded = embedded * mask
        seq_lengths = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        averaged = masked_embedded.sum(dim=1) / seq_lengths

        hidden = self.relu(self.fc1(averaged))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)

        return self.log_softmax(output)

class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, model: DeepAveragingNetwork, word_embeddings: WordEmbeddings, use_typo_correction: bool = False):
        self.model = model
        self.word_embeddings = word_embeddings
        self.device = model.device
        self.model.eval()

        self.typo_corrector = TypoCorrector(word_embeddings.word_indexer) if use_typo_correction else None
        
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        word_indices = []
        for word in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(word)
            if idx == -1 and has_typos and self.typo_corrector:
                corrected_word = self.typo_corrector.find_correction(word)
                if corrected_word:
                    idx = self.word_embeddings.word_indexer.index_of(corrected_word)

            if idx == -1:
                idx = self.word_embeddings.word_indexer.index_of("UNK")
            word_indices.append(idx)

        word_tensor = torch.LongTensor(word_indices).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.model(word_tensor)
            pred = torch.argmax(log_probs, dim=1).item()
        
        return pred


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    hidden_size = 100
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepAveragingNetwork(word_embeddings, hidden_size)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    typo_corrector = TypoCorrector(word_embeddings.word_indexer) if train_model_for_typo_setting else None

    def prepare_data(examples):
        data = []
        for ex in examples:
            word_indices = []
            for word in ex.words:
                idx = word_embeddings.word_indexer.index_of(word)
                if idx == -1 and train_model_for_typo_setting and typo_corrector:
                    corrected_word = typo_corrector.find_correction(word)
                    if corrected_word:
                        idx = word_embeddings.word_indexer.index_of(corrected_word)
                
                if idx == -1:  # Unknown word
                    idx = word_embeddings.word_indexer.index_of("UNK")
                word_indices.append(idx)
            data.append((word_indices, ex.label))
        return data
    
    train_data = prepare_data(train_exs)
    dev_data = prepare_data(dev_exs)

    model.train()
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            max_len = max(len(word_indices) for word_indices, _ in batch)
            pad_idx = word_embeddings.word_indexer.index_of("PAD")
            
            batch_inputs = []
            batch_labels = []
            
            for word_indices, label in batch:
                padded = word_indices + [pad_idx] * (max_len - len(word_indices))
                batch_inputs.append(padded)
                batch_labels.append(label)

            inputs = torch.LongTensor(batch_inputs).to(device)
            labels = torch.LongTensor(batch_labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for word_indices, label in dev_data:
                inputs = torch.LongTensor(word_indices).unsqueeze(0).to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1).item()
                
                if pred == label:
                    correct += 1
                total += 1
        
        dev_acc = correct / total
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Dev Acc: {dev_acc:.4f}")
        model.train()

    return NeuralSentimentClassifier(model, word_embeddings, use_typo_correction=train_model_for_typo_setting)

