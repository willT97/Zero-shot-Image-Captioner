import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import sys
import bcolz
from torchvision import transforms
from noc_model import EncoderCNN, DecoderRNN
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '../utils/'))
from build_vocab import Vocabulary


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_glove_dict():
    vectors = bcolz.open(f'{args.glove_path}/6B.300.dat')[:]
    words = pickle.load(open(f'{args.glove_path}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{args.glove_path}/6B.300_idx.pkl', 'rb'))

    return {w: vectors[word2idx[w]] for w in words}

def create_weights_matrix(vocab, vocab_size, embed_size, glove):
    matrix_len = vocab_size
    weights_matrix = np.zeros((matrix_len, embed_size))
    words_found = 0
    # iterate through size of vocab
    for i in range(vocab_size):
        try:
            weights_matrix[i] = glove[vocab.idx2word[i]]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_size, ))

    return weights_matrix


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    glove = create_glove_dict()
    weights_matrix = create_weights_matrix(vocab, len(vocab), args.embed_size, glove)
    weights_matrix = torch.tensor(weights_matrix).detach().cpu()

    # Build models
    encoder = EncoderCNN(args.hidden_size, len(vocab)).eval().to(device)  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, weights_matrix).to(device)
    # Load the trained model parameters
    if torch.cuda.is_available():
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))
    else:
        encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
        decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu'))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    print(image_tensor.size())

    # Generate an caption from the image
    # feature is the vocab size features
    feature = encoder(image_tensor)

    x = torch.argmax(feature)

    # Useful information to print
    #for value, indice in zip(values,indices):
    #    print("Most likely prediction: " + vocab.idx2word[int(indice)])
    #    print("Index is: " + str(int(indice)))
    #    print("With a score of: " + str(int(value)))
    # pass through start token

    start_token = torch.tensor(vocab.word2idx['<start>']).to(device)
    sampled_ids = decoder.sample(feature, start_token)
    # sample ids is the caption

    #sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='../../best_models/noc/encoder_noc.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='../../best_models/noc/decoder_noc.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--glove_path', type=str, default='/glove', help='path for glove embeddings')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
