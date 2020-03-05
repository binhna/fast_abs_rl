# from flask_restful import Api, Resource, reqparse
from flask import Flask, Blueprint, request, jsonify, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from flask_cors import CORS
import os, sys
from os.path import join
import json
from cytoolz import identity, concat, curry
from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor

from predict import decode, ARGS

#from gevent.wsgi import WSGIServer
app = Flask(__name__, static_folder='ui', template_folder='ui')
Bootstrap(app)
cors = CORS(app, resources={r'/*': {"origins": '*'}})
app.config['CORS_HEADER'] = 'Content-Type'


model_dir = '../data/model'
beam_size = 5
max_len = 30
cuda = False
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

@app.route('/summary', methods=['GET'])
def gey_summary():
    args = ARGS()

    setattr(args, 'model_dir', '../data/model')
    setattr(args, 'batch', 1)
    setattr(args, 'beam', 5)
    setattr(args, 'div', 1.0)
    setattr(args, 'max_dec_word', 30)
    setattr(args, 'cuda', False)
    setattr(args, 'extractor', extractor)
    setattr(args, 'abstractor', abstractor)

    text = request.args.get('text')
    setattr(args, 'text', text)
    result = decode(args, True)
    print(result)
    return jsonify({'summary': result})
