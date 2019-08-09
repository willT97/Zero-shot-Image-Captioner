import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
import bcolz
import gc
from noc_model import EncoderCNN, DecoderRNN
# pack padded sequence are used as input to the RNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torchvision import datasets

sys.path.insert(1, os.path.join(sys.path[0], '../../coco/PythonAPI/'))
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

sys.path.insert(1, os.path.join(sys.path[0], '../utils/'))
from data_loader import get_loader
from caption_data_loader import get_caption_loader
from build_vocab import Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# generates a caption for a given image
# returns list of strings containing each word of the caption
def generate_caption(image, encoder, decoder, vocab):

    with torch.no_grad():
        feature = encoder(image)
        start_token = torch.tensor(vocab.word2idx['<start>']).to(device)
        sampled_ids = decoder.sample(feature, start_token)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        sampled_caption.append(word)

    return sampled_caption

# generates the sentences from the test set and saves to file
# returns the F1 score
def evaluate_test_set(image_dir, test_set, encoder, decoder, vocab, transform, batch_size, num_workers, object_names, save_path):

    # create the data loader
    val_loader = get_loader(image_dir, test_set, vocab, transform, batch_size,
                            shuffle=False, num_workers=num_workers)

    # initial F1 scores
    f1_count = 0
    total_count = 0
    # to store the captions in
    results = []

    # prevents duplicates in the json file
    id_occurences = {}

    for j, (images, captions, lengths, img_ids) in enumerate(val_loader):
        for image, length, img_id in zip(images, lengths, img_ids):
            # number of captions generated
            total_count += 1

            image = image.view(1, *image.size())
            image = image.to(device)

            # calculates the caption from the image
            sampled_caption = generate_caption(image, encoder, decoder, vocab)

            # checks if any of the objects are mentioned
            # returns a positive example

            contained = False
            # checks if any of the object names are contained in
            # the sampled captions
            for word in sampled_caption:
                for name in object_names:
                    if name in word:
                        contained = True
            if contained:
                f1_count += 1

            sentence = ' '.join(sampled_caption)

            # for the general evaluation
            if str(img_id) not in id_occurences:
                results.append({"image_id": int(img_id), "caption": sentence})
                id_occurences[str(img_id)] = 1

    # write results to file
    with open(save_path, 'w') as f:
        json.dump(results, f)

    return f1_count / total_count

# loads the caption file and returns the scores
# BLEU 1,2,3,4 CIDER, METEOR, SPICE
# in a dictionary
def coco_evaluate(object_class, save_path, name):
    results = {}

    # Evaluation section
    coco = COCO(object_class)
    cocoRes = coco.loadRes(f'{name}_val_results')

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    cocoEval.evaluate()

    # write the results to the object file
    # scores = []
    #with open(os.path.join(args.results_path, f'{name}_scores.txt'), 'a') as f:
    for metric, score in cocoEval.eval.items():
        results[metric] = score
        #scores.append(score)

    return results


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

    #print(words_found)

    return weights_matrix

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # add glove word embeddings
    glove = create_glove_dict()
    weights_matrix = create_weights_matrix(vocab, len(vocab), args.embed_size, glove)
    weights_matrix = torch.tensor(weights_matrix).detach().cpu()

    # Build data loader
    if args.novel:
        # changes the caption path to the novel split
        print("Novel accepted")
        caption_path = args.split_caption_path
    else:
        caption_path = args.caption_path

    data_loader = get_loader(args.image_dir, caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)


    #test_loader = get_loader(args.val_image_dir, args.val_caption_path, vocab,
    #                         transform, args.batch_size,
    #                         shuffle=False, num_workers=args.num_workers)

    # training on images with visual features and captions
    # get each loader

    #novel_loader = get_loader(args.image_dir, args.novel_training_captions, vocab,
    #                         transform, 30,
    #                         shuffle=True, num_workers=args.num_workers)

    #bottle_loader = get_loader(args.image_dir, args.bottle_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #bus_loader = get_loader(args.image_dir, args.bus_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #couch_loader = get_loader(args.image_dir, args.couch_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #microwave_loader = get_loader(args.image_dir, args.microwave_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #pizza_loader = get_loader(args.image_dir, args.pizza_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #racket_loader = get_loader(args.image_dir, args.racket_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #suitcase_loader = get_loader(args.image_dir, args.suitcase_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)
    #zebra_loader = get_loader(args.image_dir, args.zebra_training_captions, vocab,
                             #transform, 18,
                             #shuffle=True, num_workers=args.num_workers)

    # iterate through the loaders each iteration
    #novel_loaders = [bottle_loader, bus_loader, couch_loader, microwave_loader, pizza_loader, racket_loader, suitcase_loader, zebra_loader]

    # just for the captions of novel objects for lstm
    #caption_loader = get_caption_loader(args.novel_training_captions, vocab,
    #                                    transform, 24,
    #                                    shuffle=True, num_workers = 0)

    # IMAGENET IMAGES
    #imagenet_dataset = datasets.ImageFolder(
    #    root=args.imagenet_dir,
    #    transform=  transforms.Compose([
    #                    transforms.RandomResizedCrop(args.crop_size, (0.9, 1)),
    #                    transforms.RandomHorizontalFlip(),
    #                    transforms.ToTensor(),
    #                    transforms.Normalize((0.485, 0.456, 0.406),
    #                                        (0.229, 0.224, 0.225))])


    #    )
    #imagenet_loader = torch.utils.data.DataLoader(
        #imagenet_dataset,
        #batch_size=args.imagenet_batch_size,
        #num_workers=args.num_workers,
        #shuffle=True
    #)

    # how to use
    #for batch_idx, (data, target) in enumerate(imagenet_loader):


    # Build the models
    encoder = EncoderCNN(args.hidden_size, len(vocab)).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, weights_matrix).to(device)

    # Loss and optimizer
    #visual_loss = nn.CrossEntropyLoss()
    visual_loss = torch.nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    caption_loss = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    best_scores = [0,0,0,0,0,0,0,0]
    best_score = 0


    # Train the models
    total_step = len(data_loader)

    # class to id
    # maps object to the target id
    target_object = {}
    for i_d, k in enumerate(["bottle", "bus", "couch", "microwave", "pizza", "racket", "suitcase", "zebra"]):
        target_object[i_d] = int(vocab.word2idx[k])


    class_names = {0: "bottle", 1: "bus", 2: "couch", 3:"microwave",
            4: "pizza", 5: "racket", 6: "suitcase", 7: "zebra"}


    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths, img_ids) in enumerate(data_loader):

            # COCO NOVeL IMAGES AND CAPTIONS
            # selects set of new images from each loader each iteration
            #nov_images, nov_captions, nov_lengths, nov_ids = next(iter(novel_loader))
            #nov_images = nov_images.to(device)

            #visual_targets = torch.zeros([30, len(vocab)], dtype=torch.float32).to(device)
            #targets = nov_captions[:,1:-2].clone()
            #for c, target in enumerate(targets):
                #for index in target:
                    #visual_targets[c, index] = 1

            #visual_features = encoder(nov_images)
            #image_loss = visual_loss(visual_features, visual_targets)

            #del nov_images

            # MSCOCO CAPTIONS LANGAUGE LOSS

            #novel_captions, novel_lengths, img_ids = next(iter(caption_loader))
            #novel_captions = novel_captions.to(device)

            # removes start token from the start of the caption
            #target_captions = novel_captions[:,1:].clone()
            ## reduces the caption lengths by 1
            #nov_lengths = list(map(lambda x: x-1, novel_lengths))
            #novel_targets, batch_sizes = pack_padded_sequence(target_captions, nov_lengths, batch_first=True)
            #novel_outputs = decoder(novel_captions, nov_lengths)

            #language_model_loss = caption_loss(novel_outputs, novel_targets)

            # SEPERATE CAPTION LOSS

            #imagenet_images, targets  = next(iter(imagenet_loader))
            #imagenet_images = imagenet_images.to(device)
            #visual_targets = torch.tensor([target_object[int(x)] for x in targets]).to(device)
            #print(visual_targets)
            #visual_targets = torch.zeros([args.imagenet_batch_size, len(vocab)], dtype=torch.float32).to(device)
            #for c, class_object in enumerate(targets):
            #    class_index = vocab.word2idx[class_names[int(class_object)]]
                #print(class_index)
            #    visual_targets[c, class_index] = 1
                #print(visual_targets)

            #visual_features = encoder(imagenet_images)
            # find out the targets
            #_, predicted = torch.max(visual_features, 1)
            #print(predicted)

            # Useful stuff to print
            #for predicted, actual in zip(visual_features, visual_targets):
            #    print("Visual model thought it was: " + vocab.idx2word[int(torch.argmax(predicted))])
            #    print("Actually it was: " + vocab.idx2word[int(torch.argmax(actual))])

            #for z,y in zip(visual_targets, predicted):
            #    print("Visual model thought it was: " + vocab.idx2word[int(y)])
            #    print("Actually it was: " + vocab.idx2word[int(z)])

            # add in lstm training on all of the captions
            # therefore including all of the words and the novel objects

            #image_loss = visual_loss(visual_features, visual_targets)
            #del imagenet_images

            # removes start token from the start of the caption
            target_captions = captions[:,1:].clone()
            # reduces the caption lengths by 1
            lengths = list(map(lambda x: x-1, lengths))

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)

            targets, batch_sizes = pack_padded_sequence(target_captions, lengths, batch_first=True)

            features = encoder(images)

            outputs = decoder(captions, lengths)

            packed_sequence = torch.nn.utils.rnn.PackedSequence(outputs, batch_sizes)
            unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True)


            for caption, feature in zip(unpacked, features):
                for prediction in caption:
                     prediction.add_(feature)

            outputs, original_lengths = pack_padded_sequence(unpacked, unpacked_len, batch_first=True)

            # calculating the loss
            loss = criterion(outputs, targets.to(device))

            ## loading the novel captions
            #novel_captions, novel_lengths, img_ids = next(iter(caption_loader))
            #novel_captions = novel_captions.to(device)

            ## removes start token from the start of the caption
            #target_captions = novel_captions[:,1:].clone()
            ## reduces the caption lengths by 1
            #lengths = list(map(lambda x: x-1, novel_lengths))
            #novel_targets, batch_sizes = pack_padded_sequence(target_captions, lengths, batch_first=True)
            #novel_outputs = decoder(novel_captions, lengths)

            #language_model_loss = caption_loss(novel_outputs, novel_targets)


            #final_loss = loss + image_loss
            #final_loss = loss + image_loss + language_model_loss
            #final_loss.backward()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            if i % args.log_step == 0:
                #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Imagenet_Loss: {:.4f}, Language_Loss {:.4f}, Perplexity: {:5.4f}'
                #      .format(epoch, args.num_epochs, i, total_step, loss.item(), image_loss.item(), language_model_loss.item(), np.exp(loss.item())))

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            if i % args.val_step == 0 and args.val_step > 0 and args.val == True:

                # calculate F1 scores

                object_list = [args.bottle_test, args.bus_test, args.couch_test, args.microwave_test, args.pizza_test, args.racket_test, args.suitcase_test,                                args.zebra_test]
                object_names = ['bottle', 'bus', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']

                # make eval function

                # takes in as arguements object list object names encoder decoder and returns the scores
                # seperate function that takes test set, name, encoder decoder and returns set of scores

                current_score = 0
                encoder = encoder.eval()
                final_scores = {}

                # iterate through each of the held out objects
                for object_class, name in zip(object_list, object_names):
                    # gets F1 score for the object set and saves generated captions to file
                    save_path = f'{name}_val_results'
                    f1_score = evaluate_test_set(args.val_image_dir, object_class, encoder, decoder, vocab, transform,
                                                 args.batch_size, args.num_workers, object_names, save_path)

                    #scores = coco_evaluate(object_class, save_path, name)
                    #scores['F1'] = f1_score
                    #final_scores[name] = scores

                    #print(scores['METEOR'])
                    #print(f1_score)
                    #current_score += scores['METEOR']
                    current_score += f1_score
                if current_score > best_score:
                    print("New best score, saving to file...")
                    # current model is the new best
                    best_score = current_score
                    #best_scores = list(current_scores)

                    torch.save(decoder.state_dict(), os.path.join(
                        args.model_path, 'decoder_language_only_best.ckpt'))
                    torch.save(encoder.state_dict(), os.path.join(
                        args.model_path, 'encoder_language_only_best.ckpt'))

                   # with open(os.path.join(args.model_path, "best_model_results.txt"), 'w') as f:
                   #     for name in object_names:
                   #         f.write("%s scores \n"%(name))
                   #         for metric, score in final_scores[name].items():
                   #             f.write('%s: %.3f \n'%(metric, score))

                        # at what point in training
                   #     f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f} \n'
                   #             .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                   #     f.write('Learning rate: {}, Batch size: {} \n'.format(args.learning_rate, args.batch_size))

                encoder = encoder.train()
                    # maybe change this to write the best files for each of the objects
                    #with open(os.path.join(args.model_path, 'annotations.json'), 'w') as outfile:
                    #    json.dump(results, outfile)
                    #print("saved")



        torch.save(decoder.state_dict(), os.path.join(
            args.model_path, 'decoder_caption_network_' + str(epoch) + '.ckpt'))
        torch.save(encoder.state_dict(), os.path.join(
            args.model_path, 'encoder_caption_network_' + str(epoch) + '.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_path', type=str, default='../../glove', help='path for glove embeddings')
    parser.add_argument('--model_path', type=str, default='noc_models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='../../mscoco/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../mscoco/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../../mscoco/annotations/captions_train2014.json', help='path for train annotation json file')


    parser.add_argument('--split_caption_path', type=str, default='../../mscoco/annotations/novel_split/captions_no_caption_rm_eightCluster_train2014.json',
                        help='path for the training split with the 8 held out objects')

    # held out objects 50% of the training data for each one to fine tune CNN at the end?
    parser.add_argument('--bottle_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_bottle_val_val_train2014.json', help = 'path for 50% of the training images containing bottles')
    parser.add_argument('--bus_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_bus_val_val_train2014.json', help = 'path for 50% of the training images containing bus')
    parser.add_argument('--couch_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_couch_val_val_train2014.json', help = 'path for 50% of the training images containing couch')
    parser.add_argument('--microwave_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_microwave_val_val_train2014.json', help = 'path for 50% of the training images containing microwave')
    parser.add_argument('--pizza_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_pizza_val_val_train2014.json', help = 'path for 50% of the training images containing pizza')
    parser.add_argument('--racket_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_racket_val_val_train2014.json', help = 'path for 50% of the training images containing racket')
    parser.add_argument('--suitcase_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_suitcase_val_val_train2014.json', help = 'path for 50% of the training images containing suitcase')
    parser.add_argument('--zebra_training_images', type=str, default='../../mscoco/annotations/novel_split/captions_split_set_zebra_val_val_novel2014.json', help = 'path for 50% of the training images containing zebra')


    parser.add_argument('--novel_training_captions', type=str, default='../../mscoco/annotations/held_out_train/all_novel_objects_train.json', help = 'path for captions in training dataset containing all the novel objects')
    parser.add_argument('--bottle_training_captions', type=str, default='../../mscoco/annotations/held_out_train/bottle_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--bus_training_captions', type=str, default='../../mscoco/annotations/held_out_train/bus_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--couch_training_captions', type=str, default='../../mscoco/annotations/held_out_train/couch_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--microwave_training_captions', type=str, default='../../mscoco/annotations/held_out_train/microwave_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--pizza_training_captions', type=str, default='../../mscoco/annotations/held_out_train/pizza_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--racket_training_captions', type=str, default='../../mscoco/annotations/held_out_train/racket_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--suitcase_training_captions', type=str, default='../../mscoco/annotations/held_out_train/suitcase_train.json', help = 'path for captions in training dataset containing this object')
    parser.add_argument('--zebra_training_captions', type=str, default='../../mscoco/annotations/held_out_train/zebra_train.json', help = 'path for captions in training dataset containing this object')


    parser.add_argument('--imagenet_dir', type=str, default='../../imagenet/imagenet_images', help='path for imagenet images of the heldout objects')

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

    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=6000, help='step size for saving trained models')
    parser.add_argument('--val_image_dir', type=str, default='../../mscoco/resized_val_2014', help='directory for resized val images')
    parser.add_argument('--val_caption_path', type=str, default='../../mscoco/annotations/captions_val2014.json', help='path for val annotation json file')

    parser.add_argument('--val_step', type=int , default=15000, help='step size for evaluating the model')
    parser.add_argument('--val', type=float , default=False, help='whether to evaluate or not')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--imagenet_batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--novel', type=float, default=False, help='decides whether to train on the split')

    args = parser.parse_args()
    print(args)
    main(args)
