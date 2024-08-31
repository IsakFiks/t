import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

# Load your data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Preprocess the data
def preprocess_data(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    return X, y, tokenizer, max_sequence_len, total_words

# Build the model
def build_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 200, input_length=max_sequence_len-1))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Generate text from the model
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Main execution
if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('training_data.txt')
    X, y, tokenizer, max_sequence_len, total_words = preprocess_data(data)

    # Build and train the model
    model = build_model(total_words, max_sequence_len)
    model.summary()
    history = model.fit(X, y, epochs=100, verbose=1)

    # Generate new text
    seed_text = "Chat is this"
    generated_text = generate_text(seed_text, 10, model, tokenizer, max_sequence_len)
    print("\nGenerated Text:")
    print(generated_text)
