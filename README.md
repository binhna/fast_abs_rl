# This work forked from from Yen-Chun Chen, but I modify the code so it can work with Japanese
The README below is from the original author, but there are some changes. You should read through it tho.
# Fast Abstractive Summarization-RL
This repository contains the code for their ACL 2018 paper:

*[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080)*.

## Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system.

## Decode summaries from the pretrained model
You will also need a preprocessed version of the CNN/DailyMail dataset.
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset or modifying your data.

To decode, run
```
python decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/pretrained] --beam=[beam_size] [--test/--val]
```
Options:
- beam_size: number of hypothesis for (diverse) beam search. (use beam_size > 1 to enable reranking)
  - beam_szie=1 to get greedy decoding results (rnn-ext + abs + RL)
  - beam_size=5 is used in the paper for the +rerank model (rnn-ext + abs + RL + rerank)
- test/val: decode on test/validation dataset

Next, make the reference files for evaluation:
```
python make_eval_references.py
```
and then run evaluation by:
```
python eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```

## Train your own models
The preprocess data is from *[here](https://github.com/ChenRocks/cnn-dailymail)*, but I changed some code to use mecab for word segmentation.

Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.


To re-train our best model:
1. pretrained a *word2vec* word embedding
```
python train_word2vec.py --save_dir /path/of/folder/to/save/model --dim 300 --data_dir /path/to.the/finished_files
```
Which ```/path/to.the/finished_files``` is the path after you've done the preprocess step above. It must contain 3 folder train, test and val and maybe the vocab_cnt.pkl

2. make the pseudo-labels
```
python make_extraction_labels.py
```
3. train *abstractor* and *extractor* using ML objectives
```
python train_abstractor.py --path=[path/to/abstractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
```
4. train the *full RL model*
```
python train_full_rl.py --path=[path/to/save/model] --abs_dir=[path/to/abstractor/model] --ext_dir=[path/to/extractor/model]
```
After the training finishes you will be able to run the decoding and evaluation following the instructions in the previous section.

The above will use the best hyper-parameters we used in the paper as default.
Please refer to the respective source code for options to set the hyper-parameters.

