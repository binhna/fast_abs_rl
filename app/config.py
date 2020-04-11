class Config:
    HOST = "http://0.0.0.0"
    PORT = "5555"
    URL = HOST + ":" + PORT
    model_dir_ja = '/mnt/binhna/summary/model200k'
    model_dir_en = '/mnt/binhna/summary/pretrained/new'
    beam_size = 5
    max_len = 30
    cuda = False
