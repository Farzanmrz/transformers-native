import os
import torch
from models.Encoder import Encoder
from utils.data_loading import get_dataset

def main():
    # Define the path to the dataset file
    data_path = os.path.join(os.path.dirname(__file__), '../data/small_wmt14_en_de.json')

    # Load the dataset
    dataset = get_dataset(data_path)

    # Create a vocabulary from the dataset
    vocab = set()
    for pair in dataset:
        vocab.update(pair['en'].split())
    vocab = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # Convert sentences to tensor
    def convert_to_tensor(sentences, vocab, pad_token='<PAD>'):
        max_len = max(len(sentence.split()) for sentence in sentences)
        tensor_data = []
        for sentence in sentences:
            indices = [vocab[word] for word in sentence.split()]
            indices += [vocab.get(pad_token, 0)] * (max_len - len(indices))
            tensor_data.append(indices)
        return torch.tensor(tensor_data, dtype=torch.long)

    eng_sentences = [pair['en'] for pair in dataset]
    eng_tensor = convert_to_tensor(eng_sentences, vocab)

    # Initialize the encoder
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff)

    # Perform encoding
    mask = None  # No masking as per the original paper
    encoded_output = encoder(eng_tensor, mask)

    # Display results for at least two examples
    num_examples = min(2, len(eng_sentences))
    for i in range(num_examples):
        print(f"Original Sentence: {eng_sentences[i]}")
        print(f"Original Tensor Shape: {eng_tensor[i].shape}")
        print(f"Encoded Output Shape: {encoded_output[i].shape}")
        print(f"Encoded Output: {encoded_output[i]}\n")

if __name__ == "__main__":
    main()
