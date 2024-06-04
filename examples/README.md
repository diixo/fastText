

## Генерация эмбедингов:

### Для одного слова:

```
word_vector = model.get_word_vector('example')
print(word_vector)
```

### Для целого предложения:

Для предложений FastText не имеет прямого метода, но можно усреднить эмбединги слов в предложении:

```
sentence = "This is an example sentence."
words = sentence.split()
sentence_vector = sum(model.get_word_vector(word) for word in words) / len(words)
print(sentence_vector)
```

Эти шаги помогут вам начать работу с FastText для генерации эмбедингов слов и предложений.


## Методы для получения представления предложения. Вот несколько подходов:

- Обучение модели FastText для фраз: Вместо того чтобы обучать модель на отдельных словах, можно обучить её на фразах или даже предложениях. Это менее распространённый подход, но он может быть полезен.

- Использование других моделей для предложений: Если вашей основной целью является получение эмбедингов для предложений, вы можете рассмотреть использование специализированных моделей, таких как Universal Sentence Encoder, Sentence-BERT, или даже использование трансформеров (например, BERT).

- Использование архитектур типа RNN или LSTM: Эти модели хорошо подходят для работы с последовательностями, такими как предложения. Вы можете обучить RNN или LSTM модель для получения эмбедингов предложений.

- Concatenation или pooling: Вы можете использовать методы, такие как конкатенация или пулинг (максимальный или минимальный), чтобы объединить эмбединги слов в представление предложения.

Если все же необходимо использовать FastText, то вот пример, как можно получить эмбединги предложений без усреднения:

### Пример с использованием RNN для создания эмбедингов предложений

Этот пример предполагает использование **PyTorch** для создания модели RNN, которая будет принимать эмбединги слов от FastText и генерировать эмбединги предложений:

```
import fasttext
import torch
import torch.nn as nn

# Загрузка предобученной модели FastText
model = fasttext.load_model('cc.en.300.bin')

# Пример предложения
sentence = "This is an example sentence."

# Получение эмбедингов слов
word_vectors = [model.get_word_vector(word) for word in sentence.split()]
word_vectors = torch.tensor(word_vectors)  # Преобразование в тензор PyTorch

# Определение RNN модели
class SentenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SentenceRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # Начальное скрытое состояние
        out, _ = self.rnn(x, h0)
        return out[:, -1, :]  # Последнее скрытое состояние как представление предложения

# Параметры модели
input_size = 300  # Размер эмбедингов FastText
hidden_size = 128  # Размер скрытого состояния RNN

# Создание и использование модели
sentence_rnn = SentenceRNN(input_size, hidden_size)
sentence_embedding = sentence_rnn(word_vectors.unsqueeze(0))  # Добавление размерности батча
print(sentence_embedding)

```

### Наложение TFIDF на эмбединг вместо усреднения

## Word vectors for 157 languages

We distribute pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia using fastText. 
These models were trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives. 
We also distribute three new word analogy datasets, for French, Hindi and Polish.

## Limitations and bias

Even if the training data used for this model could be characterized as fairly neutral, this model can have biased predictions.

Cosine similarity can be used to measure the similarity between two different word vectors. If two two vectors are identical, the cosine similarity will be 1. For two completely unrelated vectors, the value will be 0. If two vectors have an opposite relationship, the value will be -1.

```
import numpy as np

def cosine_similarity(word1, word2):
    return np.dot(model[word1], model[word2]) / (np.linalg.norm(model[word1]) * np.linalg.norm(model[word2]))

print(cosine_similarity("man", "boy"))

#0.061653383

print(cosine_similarity("man", "ceo"))

#0.11989131

print(cosine_similarity("woman", "ceo"))

#-0.08834904

```
