

# do power spectrum analysis; delta, theta, alpha, beta, gamma

import numpy as np
import scipy.signal as signal
import mne
import matplotlib.pyplot as plt



class EEGPowerSpectrum:
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs
        self.freqs = None
        self.psd = None
        self.delta = None
        self.theta = None
        self.alpha = None
        self.beta = None
        self.gamma = None
    
       
    def compute_power_spectrum(self):
        self.freqs, self.psd = signal.welch(self.data, fs=self.fs)
        return self.freqs, self.psd 
    
    
    def compute_band_power(self, ifNormalize=True):
        self.deltaPsd = self.psd[(self.freqs >= 0.5) & (self.freqs < 4)].sum()
        self.thetaPsd = self.psd[(self.freqs >= 4) & (self.freqs < 8)].sum()
        self.alphaPsd = self.psd[(self.freqs >= 8) & (self.freqs < 12)].sum()
        self.betaPsd = self.psd[(self.freqs >= 12) & (self.freqs < 28)].sum()
        self.gammaPsd = self.psd[(self.freqs >= 28) & (self.freqs < 40)].sum()
        self.eegPsd = self.psd[(self.freqs >= 0.5) & (self.freqs < 40)].sum()
        
        if ifNormalize:
            self.delta = self.deltaPsd / self.eegPsd
            self.theta = self.thetaPsd / self.eegPsd
            self.alpha = self.alphaPsd / self.eegPsd
            self.beta = self.betaPsd / self.eegPsd
            self.gamma = self.gammaPsd / self.eegPsd
        return self.delta, self.theta, self.alpha, self.beta, self.gamma



if __name__ == '__main__':
    sinWave = np.sin(2 * np.pi * 30 * np.linspace(0, 1, 1000))
    fs = 1000
    powerSpectrum = EEGPowerSpectrum(sinWave, fs)
    
    freqs, psd = powerSpectrum.compute_power_spectrum()
    delta, theta, alpha, beta, gamma = powerSpectrum.compute_band_power()
    # plot power spectrum
    plt.figure()
    plt.plot(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2/Hz)')
    
    # plot band power
    plt.figure()
    plt.bar(['delta', 'theta', 'alpha', 'beta', 'gamma'], [delta, theta, alpha, beta, gamma])
    plt.xlabel('Frequency band')
    plt.ylabel('Power')
    plt.show()