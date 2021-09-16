import util
import librosa
import librosa.display
from variables import *
from create_spec import spectrogram_image


def create_spectrograms_full():
    for track_id in tracks.index:
        try:
            filename = util.get_audio_path(AUDIO_DIR, track_id)
            if set_size == 'medium':
                filename = filename[:-4] + '.mp3'
            else:
                filename = filename[:-4] + '.wav'

            out = spec_full_path + tracks['set', 'split'].loc[track_id] + '/' + tracks['track',
                                                                                       'genre_top'].loc[track_id] + '/spectrogram_' + str(track_id) + '_full' + '.png'
            if tracks['track', 'genre_top'].loc[track_id] == 'Old-Time / Historic':
                out = spec_full_path + tracks['set', 'split'].loc[track_id] + '/' + \
                    'Old-Time - Historic' + '/spectrogram_' + \
                    str(track_id) + '_full' + '.png'
            x, sr = librosa.load(filename, offset=0.0, sr=None, mono=True)
            spectrogram_image(x, sr=sr, out=out, hop_length=512, n_mels=128)
        except:
            continue


if __name__ == "__main__":
    create_spectrograms_full()
