import os
import torch
import unittest
from models.layers.InputEmbeddings import InputEmbeddings
from utils.data_loading import get_dataset, pad_sequences


class TestInputEmbeddings(unittest.TestCase):

	def setUp( self ):
		base_path = os.path.abspath(os.path.dirname(__file__))
		self.data_path = os.path.join(base_path, '../../data/small_wmt14_en_de.json')
		self.dataset = get_dataset(self.data_path)

		# Create vocabulary from dataset
		self.vocab = { word: idx for idx, word in enumerate(set(' '.join([ item[ 'en' ] for item in self.dataset ]).split())) }
		self.vocab_size = len(self.vocab)
		self.d_model = 512

	def convert_to_tensor( self, sentences, vocab, pad_token = 0 ):
		max_len = max(len(sentence.split()) for sentence in sentences)
		tensor_data = [ ]
		for sentence in sentences:
			indices = [ vocab.get(word, vocab.get('<unk>', 0)) for word in sentence.split() ]
			indices += [ pad_token ] * (max_len - len(indices))
			tensor_data.append(indices)
		return torch.tensor(tensor_data, dtype = torch.long)

	def test_create_position_embedding( self ):
		input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)
		pos_embedding = input_embeddings.create_position_embedding(self.d_model, 10)
		self.assertEqual(pos_embedding.shape, (1, 10, self.d_model), "Position embedding shape mismatch")

	def test_forward( self ):
		input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)
		eng_sentences = [ item[ 'en' ] for item in self.dataset ]
		eng_tensor = self.convert_to_tensor(eng_sentences, self.vocab)

		# Ensure the tensor is on the correct device
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		eng_tensor = eng_tensor.to(device)
		input_embeddings = input_embeddings.to(device)

		eng_embeddings = input_embeddings(eng_tensor)
		self.assertEqual(eng_embeddings.shape, (eng_tensor.shape[ 0 ], eng_tensor.shape[ 1 ], self.d_model), "Embeddings shape mismatch")

	def test_padding( self ):
		input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)
		sentences = [ "this is a test", "this is another test that is longer" ]
		tensor = self.convert_to_tensor(sentences, self.vocab)

		# Ensure padding works as expected
		expected_length = max(len(s.split()) for s in sentences)
		for seq in tensor:
			self.assertEqual(len(seq), expected_length, "Padding did not work correctly")

	def test_edge_case_empty_input( self ):
		input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)
		tensor = self.convert_to_tensor([ "" ], self.vocab)

		# Ensure it handles empty input without error
		embeddings = input_embeddings(tensor)
		self.assertEqual(embeddings.shape, (1, 0, self.d_model), "Empty input case failed")

	def test_edge_case_single_word( self ):
		input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)
		tensor = self.convert_to_tensor([ "test" ], self.vocab)

		# Ensure it handles single word input correctly
		embeddings = input_embeddings(tensor)
		self.assertEqual(embeddings.shape, (1, 1, self.d_model), "Single word case failed")


if __name__ == '__main__':
	unittest.main()
