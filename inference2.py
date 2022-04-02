import yaml
from model import AE
import torch
import pickle
import torch
import torch.nn as nn
import scipy
from data.utils import melspectrogram2wav
from scipy.io.wavfile import write
from data.utils import get_spectrograms
import torch.nn.functional as F

def from_wav(path, m, s):
    mel, _ = get_spectrograms(path)
    return mel
    #return (mel - m) / s

def save_wav (M, path):
    with open("data/LibriTTS2/attr.pkl", 'rb') as f:
        attr = pickle.load(f)
    M = M * attr['std'] + attr['mean']
    wav_data = melspectrogram2wav(M)
    write(path, rate=24000, data=wav_data)
    
    '''recov = librosa.feature.inverse.mel_to_audio (M=M, 
        sr=hp.sr, 
        n_fft=hp.n_fft, 
        hop_length=hp.hop_length, 
        win_length=hp.win_length, 
        power=hp.power, 
        n_iter=hp.n_iter)
    
    recov *= 32767 / max (0.01, np.max(np.abs(recov)))
    scipy.io.wavfile.write (path, 16000, recov.astype(np.int16))'''

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
model = AE(config)
model.load_state_dict(torch.load(f'saved_models3/model.ckpt'))
#model.load_state_dict(torch.load(f'best_model/vctk_model.ckpt'))
model.eval()

segment_size = 128

with open("data/LibriTTS2/reduced_train_128.pkl", 'rb') as f:
    data = pickle.load(f)

'''# woman to same woman
k1 = "1988_147956_000008_000000.wav"
k2 = "1988_24833_000058_000002.wav"

# woman to man
k1 = "1988_147956_000008_000000.wav"
k2 = "1272_128104_000005_000014.wav"

# woman to other woman
k1 = "1988_147956_000008_000000.wav"
k2 = "5895_34615_000029_000002.wav"

# man to other man
k1 = "2902_9008_000054_000002.wav"
k2 = "3000_15664_000013_000003.wav"'''

ind = 0

if ind==0:
    # man to woman
    k1 = "6836_61804_000026_000000.wav"
    k2 = "7517_100429_000002_000001.wav"

if ind==1:
    # woman to woman
    k1 = "4051_11217_000021_000000.wav"
    k2 = "7517_100429_000002_000001.wav"

print(data[k1].shape, data[k2].shape)

v1 = data[k1]
n1 = v1.shape[0]//segment_size
v1 = v1[:n1*segment_size]
v2 = data[k2]
n2 = v2.shape[0]//segment_size
v2 = v2[:n2*segment_size]

'''with open("data/LibriTTS/attr.pkl", 'rb') as f:
    attr = pickle.load(f)
    m, s = attr['std'], attr['mean']

k1 = "/home/nadavru/Downloads/recording1.wav"
k2 = "/home/nadavru/Downloads/ofri_recording.wav"

v1 = from_wav(k1, m, s)
v2 = from_wav(k2, m, s)

assert v1.shape[0]>=segment_size
assert v2.shape[0]>=segment_size

v1 = v1[:segment_size]
v2 = v2[:segment_size]'''


save_wav(v1, "results/source.wav")
save_wav(v2, "results/target.wav")

v1 = torch.Tensor(v1).reshape(n1,segment_size,-1).transpose(1, 2)
v2 = torch.Tensor(v2).reshape(n2,segment_size,-1).transpose(1, 2)

'''#dec = model.inference(v1,v1)
emb = model.speaker_encoder(v1)
mu1, _ = model.content_encoder(v1)
mu2, _ = model.content_encoder(v2)
#mu += torch.randn((1,128,38))
#plusminus = torch.Tensor([-1,1]*128*19).reshape((1,128,38))
mu = mu2
dec = model.decoder(mu, emb)'''

with torch.no_grad():

    #dec = model.inference(v1,v2)
    emb1 = model.speaker_encoder(v1).median(dim=0, keepdim=True)
    emb2 = model.speaker_encoder(v2).median(dim=0, keepdim=True)
    mu1, _ = model.content_encoder(v1)
    mu2, _ = model.content_encoder(v2)
    mu1 = mu1[:1]
    mu2 = mu2[:1]

    dec = model.decoder(mu2, emb1)
    save_wav(dec.transpose(1, 2).squeeze(0).detach().numpy(), "results/target2source.wav")

    dec = model.decoder(mu1, emb2)
    save_wav(dec.transpose(1, 2).squeeze(0).detach().numpy(), "results/source2target.wav")

    dec = model.decoder(mu1, emb1)
    save_wav(dec.transpose(1, 2).squeeze(0).detach().numpy(), "results/rec_source.wav")

    dec = model.decoder(mu2, emb2)
    save_wav(dec.transpose(1, 2).squeeze(0).detach().numpy(), "results/rec_target.wav")

    '''for i in range(1,10):
        dec = model.decoder(mu, (0.99+i/1000)*emb2+(0.01-i/1000)*emb1)
        save_wav(dec.transpose(1, 2).squeeze(0).detach().numpy(), f"results/source2target_{i}.wav")
        print(i, F.l1_loss(model.speaker_encoder(dec), emb2))'''
    exit()

    print(dec.shape)

    criterion = nn.L1Loss()
    loss_rec = criterion(dec, v)
    print(loss_rec)
