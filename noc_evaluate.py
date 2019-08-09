import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import sys
import bcolz
import json
from torchvision import transforms
from noc_model import EncoderCNN, DecoderRNN
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '../utils/'))
from build_vocab import Vocabulary

sys.path.insert(1, os.path.join(sys.path[0], '../../coco/PythonAPI/'))
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

sys.path.insert(1, os.path.join(sys.path[0], '../utils/'))
from data_loader import get_loader
from build_vocab import Vocabulary


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# add the data loaders for all of the images
# add all dataloaders to list to iterate through to get the scores
# perform evaluation the same way as previously
# put the results to a file


def create_glove_dict():
    vectors = bcolz.open(f'{args.glove_path}/6B.300.dat')[:]
    words = pickle.load(open(f'{args.glove_path}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{args.glove_path}/6B.300_idx.pkl', 'rb'))

    return {w: vectors[word2idx[w]] for w in words}

def create_weights_matrix(vocab, vocab_size, embed_size, glove):
    matrix_len = vocab_size
    weights_matrix = np.zeros((matrix_len, embed_size))
    words_found = 0
    #print(vocab_size)
    # iterate through size of vocab
    for i in range(vocab_size):
        try:
            weights_matrix[i] = glove[vocab.idx2word[i]]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_size, ))
            #print(len(weights_matrix[i]))

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
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    glove = create_glove_dict()
    weights_matrix = create_weights_matrix(vocab, len(vocab), args.embed_size, glove)
    weights_matrix = torch.tensor(weights_matrix).detach().cpu()

    # load the evaluation data
    print(args.encoder_path)
    print(args.decoder_path)

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

    # perform evaluation here

    object_list = [args.bottle_test, args.bus_test, args.couch_test, args.microwave_test, args.pizza_test, args.racket_test, args.suitcase_test, args.zebra_test]
    object_names = ['bottle', 'bus', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']

    # iterate through each of the held out objects
    for object_class, name in zip(object_list, object_names):

        data_loader = get_loader(args.image_dir, object_class, vocab,
                                transform, args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        f1_count = 0
        total_count = 0
        results = []
        id_occurences = {}

        # gets the results into a json file for the captions
        # Sums up the F1 scores and the total number of captions
        for j, (images, captions, lengths, img_ids) in enumerate(data_loader):
            for image, length, img_id in zip(images, lengths, img_ids):
                print(j)
                total_count += 1
                image = image.view(1, *image.size())
                image = image.to(device)
                #print(image.size())

                with torch.no_grad():
                    feature = encoder(image)

                    start_token = torch.tensor(vocab.word2idx['<start>']).to(device)
                    sampled_ids = decoder.sample(feature, start_token)

                #sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

                # Convert word_ids to words
                sampled_caption = []
                for word_id in sampled_ids:
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    sampled_caption.append(word)

                contained = False
                for word in sampled_caption:
                    if word in object_names:
                        contained = True
                if contained:
                    f1_count += 1

                # removes start token
                # not needed for parallel captioner
                # sentence = ' '.join(sampled_caption[1:])
                sentence = ' '.join(sampled_caption)

                # for the general evaluation
                if str(img_id) not in id_occurences:
                    results.append({"image_id": int(img_id), "caption": sentence})
                    id_occurences[str(img_id)] = 1
        #name = 'alexnet'
        with open(f'{name}_val_results', 'w') as outfile:
            json.dump(results, outfile)
        print("saved")

        # Evaluation section
        coco = COCO(object_class)
        cocoRes = coco.loadRes(f'{name}_val_results')

        # create cocoEval object by taking coco and cocoRes
        cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

        # please remove this line when evaluating the full validation set
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        cocoEval.evaluate()

        # write the results to the object file

        with open(os.path.join(args.results_path, f'{name}_scores.txt'), 'a') as f:
            for metric, score in cocoEval.eval.items():
                f.write('%s: %.3f \n'%(metric, score))
            f.write('F1: %.3f' %(f1_count/total_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../../mscoco/resized_val_2014/', help='path for the validation images')
    parser.add_argument('--encoder_path', type=str, default='noc_models/encoder_caption_network_1.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='noc_models/decoder_caption_network_1.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='../../mscoco/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--glove_path', type=str, default='../../glove', help='path for glove embeddings')
    parser.add_argument('--results_path', type=str, default='results/', help='path for the results file')
    parser.add_argument('--batch_size', type=int, default=34)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=224)

    parser.add_argument('--val_2014', type=str, default='../../mscoco/annotations/captions_val2014.json', help='path for vocabulary wrapper')


    # annotations for each of 8 held out objects and validation scripts

    # bottle
    parser.add_argument('--bottle_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_bottle_val_test_novel2014.json', help='path for vocabulary wrapper')
    # bus
    parser.add_argument('--bus_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_bus_val_test_novel2014.json', help='path for vocabulary wrapper')
    # couch
    parser.add_argument('--couch_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_couch_val_test_novel2014.json', help='path for vocabulary wrapper')
    # microwave
    parser.add_argument('--microwave_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_microwave_val_test_novel2014.json', help='path for vocabulary wrapper')
    # pizza
    parser.add_argument('--pizza_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_pizza_val_test_novel2014.json', help='path for vocabulary wrapper')
    # racket
    parser.add_argument('--racket_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_racket_val_test_novel2014.json', help='path for vocabulary wrapper')
    # suitcase
    parser.add_argument('--suitcase_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_suitcase_val_test_novel2014.json', help='path for vocabulary wrapper')
    # zebra
    parser.add_argument('--zebra_test', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_zebra_val_test_novel2014.json', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)

