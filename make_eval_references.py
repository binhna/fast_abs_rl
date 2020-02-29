""" make reference text files needed for ROUGE evaluation """
import json
import argparse
import os
from os.path import join, exists
from time import time
from datetime import timedelta

from utils import count_data
from decoding import make_html_safe

# try:
#     DATA_DIR = os.environ['DATA']
# except KeyError:
#     print('please use environment variable to specify data directories')


def dump(args):
    start = time()
    print('start processing {} split...'.format(args.mode))
    data_dir = join(args.data_dir, args.mode)
    dump_dir = join(args.data_dir, 'refs', args.mode)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents = data['abstract']
        with open(join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def main(args):
    for split in ['val', 'test']:  # evaluation of train data takes too long
        if not exists(join(args.data_dir, 'refs', split)):
            os.makedirs(join(args.data_dir, 'refs', split))
        setattr(args, 'mode', split)
        dump(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--data_dir', required=True,
                        help='path data which contains train, val, test folders and vocab_cnt.pkl')
    args = parser.parse_args()
    # args.cuda = torch.cuda.is_available() and not args.no_cuda
    main(args)
