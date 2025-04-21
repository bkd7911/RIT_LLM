import numpy as np
import torch
import matplotlib.pyplot as plt
def MySTFT(wav, sr, nfft, hop, window, pad):
  n = int(np.floor(len(wav[0])/hop))
  while n*hop+len(window)>len(wav[0]):
    n = n-1
  sp = []
  for s in range(n+1):
    sp.append(torch.fft.fft(window*wav[0,s*hop:s*hop+len(window)]))
  return torch.stack(sp).T[:int(np.floor(nfft/2))+1,:]

def iMySTFT(wav, sr, nfft, hop, window, pad):
  n = int(np.floor(len(wav[0])/hop))
  while n*hop+len(window)>len(wav[0]):
    n = n-1
  sp = []
  for s in range(n+1):
    sp.append(torch.fft.fft(window*wav[0,s*hop:s*hop+len(window)]))
  rw = torch.zeros(1,n*hop+len(window))
  for s in range(n-1):
    rw[0,s*hop:s*hop+len(window)] += torch.real(torch.fft.ifft(sp[s]))
  return rw #torch.stack(sp).T[:int(np.floor(nfft/2))+1,:]