# from flask_restful import Api, Resource, reqparse
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)

from flask import Flask, Blueprint, request, jsonify, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from app.config import Config
from flask_cors import CORS
import os, sys
from os.path import join
from datetime import datetime
import json
from langdetect import detect
from cytoolz import identity, concat, curry
from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor

from predict import decode, ARGS

#from gevent.wsgi import WSGIServer
app = Flask(__name__, static_folder='static', template_folder='templates')
Bootstrap(app)
cors = CORS(app, resources={r'/*': {"origins": '*'}})
app.config['CORS_HEADER'] = 'Content-Type'


model_dir_ja = Config.model_dir_ja
model_dir_en = Config.model_dir_en
beam_size = Config.beam_size
max_len = Config.max_len
cuda = Config.cuda

with open(join(model_dir_ja, 'meta.json')) as f:
    meta_ja = json.loads(f.read())
if meta_ja['net_args']['abstractor'] is None:
    # NOTE: if no abstractor is provided then
    #       the whole model would be extractive summarization
    assert beam_size == 1
    abstractor_ja = identity
else:
    if beam_size == 1:
        abstractor_ja = Abstractor(join(model_dir_ja, 'abstractor'),
                                max_len, cuda)
    else:
        abstractor_ja = BeamAbstractor(join(model_dir_ja, 'abstractor'),
                                    max_len, cuda)
extractor_ja = RLExtractor(model_dir_ja, cuda=cuda)


with open(join(model_dir_en, 'meta.json')) as f:
    meta_en = json.loads(f.read())
if meta_en['net_args']['abstractor'] is None:
    # NOTE: if no abstractor is provided then
    #       the whole model would be extractive summarization
    assert beam_size == 1
    abstractor_en = identity
else:
    if beam_size == 1:
        abstractor_en = Abstractor(join(model_dir_en, 'abstractor'),
                                max_len, cuda)
    else:
        abstractor_en = BeamAbstractor(join(model_dir_en, 'abstractor'),
                                    max_len, cuda)
extractor_en = RLExtractor(model_dir_en, cuda=cuda)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route("/")
@app.route("/home")
def index():
    return render_template('index.html',
                           title='Text Summarization',
                           url=Config.URL)

@app.route('/summary', methods=['POST'])
def gey_summary():
    args = ARGS()
    
    text = request.form['text']
    lang = detect(text)
    print('Language: ', lang)
    if lang == 'ja':
        model_dir = model_dir_ja
        extractor = extractor_ja
        abstractor = abstractor_ja
    else:
        model_dir = model_dir_en
        extractor = extractor_en
        abstractor = abstractor_en
    setattr(args, 'model_dir', model_dir)
    setattr(args, 'batch', 1)
    setattr(args, 'beam', beam_size)
    setattr(args, 'div', 1.0)
    setattr(args, 'max_dec_word', 30)
    setattr(args, 'cuda', cuda)
    setattr(args, 'extractor', extractor)
    setattr(args, 'abstractor', abstractor)
    #text = request.form['text']
    #print(text)
    
    setattr(args, 'text', text)
    text, result, source_text = decode(args, True)
    print(source_text)

    #print(type(result))
    if lang == 'ja':
        result = result.replace(' ', '')
    else:
        result = result.split('\n\n')
        for i, r in enumerate(result):
            r = r[0].upper() + r[1:]
            result[i] = r
        result = '\n\n'.join(result)
    #print(result)
    with open('/mnt/binhna/summary/log.txt', 'a') as f:
        f.write(f"\n\n======================={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}=======================\n")
        f.write(text)
        f.write("\n=============================================================================================\n")
        f.write(result)
    return jsonify({'summary': result, 'highlight': source_text})
