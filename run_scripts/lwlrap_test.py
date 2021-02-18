import torch
import librosa

from model.metrics import Lwlrap


mels = librosa.mel_frequencies(n_mels=256, fmin=50, fmax=15000)

s1 = torch.tensor([[0.5, 0.3, 0.9, 0.1]]), torch.tensor([[0, 0, 1, 0]])
s2 = torch.tensor([[0.5, 0.7, 0.9, 0.1]]), torch.tensor([[0, 1, 0, 0]])

lwlrap = Lwlrap(None)
lwlrap.update({'prediction': s1[0], 'target': s1[1]})
lwlrap.update({'prediction': s2[0], 'target': s2[1]})

print(lwlrap.compute())