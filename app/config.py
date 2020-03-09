class Config:
    HOST = "http://0.0.0.0"
    PORT = "5555"
    URL = HOST + ":" + PORT
    model_dir = '../data/model'
    beam_size = 5
    max_len = 30
    cuda = False
