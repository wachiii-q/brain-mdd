import numpy as np
import scipy.signal as signal
import mne
import matplotlib.pyplot as plt



class EEGPowerSpectrum:
    def __init__(self, data, fs, ifNormalize=True):
        """        
        :param data: EEG data of shape (n_epochs, n_channels, n_samples)
        :param fs: Sampling frequency in Hz
        :param ifNormalize: Normalize band power by total power
        """
        self.data = data
        self.fs = fs
        self.freqs = None
        self.psd = None
        self.delta = None
        self.theta = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.isNormalized = ifNormalize
    
    def compute_power_spectrum(self, data):
        self.freqs, self.psd = signal.welch(data, fs=self.fs, nperseg=self.fs, noverlap=self.fs / 2)
    
    def compute_band_power(self):
        self.delta = self.psd[(self.freqs >= 0.5) & (self.freqs < 4)].sum()
        self.theta = self.psd[(self.freqs >= 4) & (self.freqs < 8)].sum()
        self.alpha = self.psd[(self.freqs >= 8) & (self.freqs < 12)].sum()
        self.beta = self.psd[(self.freqs >= 12) & (self.freqs < 28)].sum()
        self.gamma = self.psd[(self.freqs >= 28) & (self.freqs < 40)].sum()
        self.eeg = self.psd[(self.freqs >= 0.5) & (self.freqs < 40)].sum()
        
        if self.isNormalized:
            self.delta = self.delta / self.eeg
            self.theta = self.theta / self.eeg
            self.alpha = self.alpha / self.eeg
            self.beta = self.beta / self.eeg
            self.gamma = self.gamma / self.eeg

    def run(self):
        # expected input.shape = (n_epochs, n_channels, n_samples)
        results = []
        for epoch in self.data:
            eachEpoch = []
            for channel in epoch:
                eegData = channel
                self.compute_power_spectrum(eegData)
                self.compute_band_power()
                eachEpoch.append([self.delta, self.theta, self.alpha, self.beta, self.gamma])
            results.append(eachEpoch)
        results = np.array(results)
        results = np.transpose(results, (0, 2, 1))    
        return np.array(results)


if __name__ == '__main__':
    sinWave = np.sin(2 * np.pi * 27 * np.linspace(0, 1, 1000))
    fs = 1000
    data = np.array([sinWave])
    data = np.expand_dims(data, axis=0)
    
    eegPowerSpectrum = EEGSubPowerSpectrum(data, fs)
    results = eegPowerSpectrum.run()    
    print(results)
    print(results.shape)
    
    # plot power spectrum   
    plt.figure()
    plt.plot(eegPowerSpectrum.freqs, eegPowerSpectrum.psd)
    plt.title('Power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()
    
    # scatter plot normalized band power
    plt.figure()
    plt.scatter(np.arange(5), results[0, 0, :])
    plt.title('Normalized band power')
    plt.xlabel('Band')
    plt.ylabel('Power')
    plt.show()
    