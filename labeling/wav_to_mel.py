import librosa
import torch
import nnAudio.Spectrogram

class FFT_parameters:
    sample_rate = 16000
    window_size = 400
    n_fft = 400
    hop_size = 160
    n_mels = 80
    f_min = 50
    f_max = 8000

def wav_to_mel(path: str):
    
    prms = FFT_parameters()
    to_spec = nnAudio.Spectrogram.MelSpectrogram(
            sr=prms.sample_rate,
            n_fft=prms.n_fft,
            win_length=prms.window_size,
            hop_length=prms.hop_size,
            n_mels=prms.n_mels,
            fmin=prms.f_min,
            fmax=prms.f_max,
            center=True,
            power=2,
            verbose=False,
            )

    wav, ori_sr = librosa.load(path, mono=True, sr=prms.sample_rate)
    lms = to_spec(torch.tensor(wav))
    lms = (lms + torch.finfo().eps).log()
    return lms
