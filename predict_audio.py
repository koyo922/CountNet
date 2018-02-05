# IDEA找不到，其实能跑(在项目属性页中配置好sys.path后才能识别，直接add to sources root 时好时坏不一定生效)
from parallel_util import parallel_process

import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd
import soundfile as sf
import keras
import sklearn
import librosa
from keras.preprocessing.sequence import pad_sequences

eps = np.finfo(np.float).eps


# noinspection PyShadowingNames
def load_model(models_path):
    # load model
    model = keras.models.load_model(models_path)
    # print model configuration
    model.summary(print_fn=logging.debug)
    return model


# noinspection PyShadowingNames
def load_scaler(scaler_params_path):
    # load standardisation parameters
    scaler = sklearn.preprocessing.StandardScaler(scaler_params_path)
    with np.load(scaler_params_path) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']
    return scaler


# noinspection PyShadowingNames
def read_in_wav(wav_path, model, scaler):
    # compute audio
    audio, rate = sf.read(wav_path, always_2d=True)
    audio = pad_sequences([audio], maxlen=80000, dtype=np.float64, padding='post')[0]
    # down_mix to mono
    audio = np.mean(audio, axis=1)
    # max normalise output
    audio /= np.max(audio, axis=0)
    # compute STFT
    x = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T
    # apply standardization
    x = scaler.transform(x)
    # cut to input shape length (500 frames x 201 STFT bins)
    x = x[:model.input_shape[1], :]
    # apply normalization
    theta = np.linalg.norm(x, axis=1) + eps
    x /= np.mean(theta)

    return x


if __name__ == '__main__':
    # noinspection PyArgumentList
    parser = argparse.ArgumentParser(description='Load Keras model and predict speaker count')
    parser.add_argument('--audio', help='audio file (16 kHz) of 5 seconds duration', default='examples/1.wav_trim.wav')
    parser.add_argument('--df', default='./res/4200f5d1_df_kid_countnet.pkl')
    parser.add_argument('-l', '--log_level', default='INFO')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    scaler = load_scaler('models/scaler.npz')
    model = load_model('models/RNN_keras2.h5')

    if args.df is None:
        x = read_in_wav(args.audio, model, scaler)
        x = x[np.newaxis, ...]  # add sample dimension
    else:
        df = pd.read_pickle(args.df).head(5)
        read_in_wav = partial(read_in_wav, model=model, scaler=scaler)
        x = parallel_process(df.dbPath.values, read_in_wav)

    y = model.predict(x, verbose=0)  # predict output
    logging.info("Speaker Count Estimate: %s", np.argmax(y, axis=1))
