"""
@Project   : seq2seq_model
@Module    : seq_to_seq_implementation.py
@Author    : Deco [deco@cubee.com]
@Created   : 11/16/17 1:19 AM
@Desc      : 
"""

import numpy as np
import time
import os
import re

import helper
import config

source_path = 'data/train.enc'
target_path = 'data/train.dec'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)

print(source_sentences[:500].split('\n'))
print(target_sentences[:500].split('\n'))

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH,
                            'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with open(in_path, 'rb') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'wb') as f:
        f.write(('<pad>' + '\n').encode())
        f.write(('<unk>' + '\n').encode())
        f.write(('<s>' + '\n').encode())
        f.write(('<\s>' + '\n').encode())
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                with open('config.py', 'ab') as cf:
                    if filename[-3:] == 'enc':
                        cf.write(('ENC_VOCAB = ' + str(index) + '\n').encode())
                    else:
                        cf.write(('DEC_VOCAB = ' + str(index) + '\n').encode())
                break
            f.write((word + '\n').encode())
            index += 1


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = line.decode('utf-8')
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'rb')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'wb')

    lines = in_file.read().decode('utf-8').splitlines()
    for line in lines:
        if mode == 'dec':  # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write((' '.join(str(id_) for id_ in ids) + '\n').encode())


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>'])
            for token in basic_tokenizer(line.encode())]


def extract_word_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    # set_words = set([word for line in data.split('\n') for word in line])
    set_words = set([word for word in data.split('\n')])
    int_to_vocab = {word_i: word for word_i, word
                    in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


source_path2 = 'data/vocab.enc'
target_path2 = 'data/vocab.dec'

source_sentences2 = helper.load_data(source_path2)
target_sentences2 = helper.load_data(target_path2)

# Build int2letter and letter2int dicts
source_int_to_letter, source_letter_to_int = extract_word_vocab(source_sentences2)
target_int_to_letter, target_letter_to_int = extract_word_vocab(target_sentences2)

print(type(source_letter_to_int.items()))
print(type(target_letter_to_int.items()))

print(list(source_letter_to_int.items())[0:10])
print(list(target_letter_to_int.items())[0:10])


if __name__ == '__main__':
    # process_data()
    pass
