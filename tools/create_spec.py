import os
import util
import librosa
import numpy
import skimage.io
from variables import *
from pathlib import Path


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X-X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    if y.size < hop_length:
        hop_length = y.size
    mels = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9)
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)
    img = 255-img
    path = Path(out)
    try:
        skimage.io.imsave(out, img)
    except:
        os.makedirs(path.parent)
        skimage.io.imsave(out, img)


def create_spectrograms():
    for track_id in tracks.index:
        try:
            filename = util.get_audio_path(AUDIO_DIR, track_id)
            if set_size == 'medium':
                filename = filename[:-4] + '.mp3'  # '.wav'
            else:
                filename = filename[:-4] + '.mp3'

            x, sr = librosa.load(filename, offset=0.0, sr=None, mono=True)
            for i in range(int(x.size / sr / 3.)):
                out = spec_path + tracks['set', 'split'].loc[track_id] + '/' + tracks['track',
                                                                                      'genre_top'].loc[track_id] + '/spectrogram_' + str(track_id) + '_' + str(i) + '.png'
                if tracks['track', 'genre_top'].loc[track_id] == 'Old-Time / Historic':
                    out = spec_path + tracks['set', 'split'].loc[track_id] + '/' + \
                        'Old-Time - Historic' + '/spectrogram_' + \
                        str(track_id) + '_' + str(i) + '.png'
                lower = 3*i*sr
                if lower + 3*sr > x.size:
                    upper = x.size
                else:
                    upper = lower + 3*sr
                spectrogram_image(x[3*i*sr:upper], sr=sr,
                                  out=out, hop_length=512, n_mels=128)
        except:
            continue


def create_spec_gtzan():
    path = "D:/gtzan/"
    for genre in os.listdir(path):
        j = 0
        for file in os.listdir(path+genre+"/"):
            x, sr = librosa.load(path+genre+"/"+file,
                                 offset=0.0, sr=None, mono=True)
            for i in range(int(x.size/sr/3.)):
                lower = 3*i*sr
                if lower + 3*sr > x.size:
                    upper = x.size
                else:
                    upper = lower + 3*sr
                if (j % 10 == 0):
                    out = "D:/gtzan_spec/test/" + genre + \
                        "/" + str(file) + str(i) + '.png'
                else:
                    out = "D:/gtzan_spec/train/" + genre + \
                        "/" + str(file) + str(i) + '.png'
                spectrogram_image(x[3*i*sr:upper], sr=sr,
                                  out=out, hop_length=512, n_mels=128)
            j += 1


if __name__ == "__main__":
    create_spectrograms()
    # create_spec_gtzan()
