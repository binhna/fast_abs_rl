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

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe


def decode(args):
    save_path = args.path
    model_dir = args.model_dir
    batch_size = args.batch
    beam_size = args.beam
    diverse = args.div
    max_len = args.max_dec_word
    cuda = args.cuda
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(args)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    if not os.path.exists(join(save_path, 'output')):
        os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = args.mode
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
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
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                decoded_sents = decoded_sents[:20]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')
    print()

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')
    parser.add_argument('--data_dir', required=True,
                        help='path data which contains train, val, test folders and vocab_cnt.pkl')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    setattr(args, 'mode', data_split)
    decode(args)
