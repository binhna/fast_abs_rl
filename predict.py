""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry
import re
from bs4 import BeautifulSoup

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe
from make_datafiles import mecab_tokenizer

class ARGS():
    def __init__(self):
        self.model_dir = ''

def clean_and_split(text):
    pattern = '(var.+?;)|(document.+?;)|(出典.+?\\n)|(出典.+?$)|(top image)|(\(全\d枚\))|(写真拡大 (全\d枚))'
    text = BeautifulSoup(text.replace('<!--3-->', '').strip(), "lxml").text
    source = re.sub('(\"\\n\")|(\'\\n\')|(\｜)', '', text)
    source = re.sub(pattern, '', source)
    source = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', source)
    source = re.sub('(\\n)+', '\n', source)
    source = re.sub('(。)', '。\n', source)
    #source = re.sub('(。)', '\n', source)
    text = source
    text = text.split('\n')
    text = [line for line in text if line]
    return text



def decode(args, predict=False):
    # save_path = args.path
    batch_size = args.batch
    beam_size = args.beam
    diverse = args.div
    start = time()
    extractor = args.extractor
    abstractor = args.abstractor
    # setup model
    text = ''
    

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    if not predict:
        dataset = DecodeDataset(args)

        n_data = len(dataset)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4,
            collate_fn=coll
        )
    else:
        n_data = 1
        loader = clean_and_split(args.text)
        loader = [[[' '.join(mecab_tokenizer(line)) for line in loader]]]
        text = '\n'.join(loader[0][0])
        
    i = 0
    #print(text)
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
            if beam_size > 1:
                #print(ext_arts)
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug
            source_text = [''.join(sent) for sent in ext_arts]
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                decoded_sents = decoded_sents[:20]
                # with open(join(save_path, 'output/{}.dec'.format(i)),
                #           'w') as f:
                #     f.write(make_html_safe('\n'.join(decoded_sents)))
                result = make_html_safe('\n\n'.join(decoded_sents))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')
    print()
    return text, result, source_text

_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))


def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))


def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beam_length = len(beams)
    # summaries with > 20 sentences lead to memory problems
    if beam_length > 20:
        beams = beams[:20]
        beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
        best_hyps = max(product(*beams), key=_compute_score)

        i = 0
        dec_outs = []
        for h in best_hyps:
            if i > beam_length:
                dec_outs.append(['NULL'])
            else:
                dec_outs.append(h.sequence)
            i += 1
        return dec_outs

    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='run decoding of the full model (RL)')
#     # parser.add_argument('--path', required=True, help='path to store/eval')
#     parser.add_argument('--model_dir', help='root of the full model')

#     # dataset split
#     data = parser.add_mutually_exclusive_group(required=True)
#     # data.add_argument('--val', action='store_true', help='use validation set')
#     # data.add_argument('--test', action='store_true', help='use test set')
#     # parser.add_argument('--data_dir', required=True,
#                         # help='path data which contains train, val, test folders and vocab_cnt.pkl')

#     # decode options
#     parser.add_argument('--batch', type=int, action='store', default=32,
#                         help='batch size of faster decoding')
#     parser.add_argument('--beam', type=int, action='store', default=1,
#                         help='beam size for beam-search (reranking included)')
#     parser.add_argument('--div', type=float, action='store', default=1.0,
#                         help='diverse ratio for the diverse beam-search')
#     parser.add_argument('--max_dec_word', type=int, action='store', default=30,
#                         help='maximun words to be decoded for the abstractor')

#     parser.add_argument('--no-cuda', action='store_true',
#                         help='disable GPU training')
#     data.add_argument('--text', type=str, help='article to be summary')
#     args = parser.parse_args()
#     args.cuda = torch.cuda.is_available() and not args.no_cuda

#     # data_split = 'test' if args.test else 'val'
#     # setattr(args, 'mode', data_split)
#     decode(args, True)
