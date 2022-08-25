import numpy as np
import numpy
import scipy.io.wavfile
import math
from scipy.fftpack import dct


def getMelEnergies(wav_file, start=0, end=0, segment_signal=False, NFFT=512, nfilt=40, frameAdmission=False):
    """
    Code adapted from http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    Accessed on August 25, 2022
    """ 
    
    sample_rate, signal = scipy.io.wavfile.read(wav_file)
    
    if segment_signal:
        if start >= end: return None
        endIndex = int(end * sample_rate)
        if endIndex >= len(signal): endIndex = len(signal)-1
        signal = signal[0:endIndex]  # Trim segment after end
        
        startIndex = int(start * sample_rate)
        if startIndex >= len(signal): startIndex = len(signal)-1
        signal = signal[int(start * sample_rate):]  # Trim segment before start
        
    signal_length = len(signal)
    
    if signal_length < NFFT: # If smaller than this would't be possible to compute FFT
        return None
    
    frame_size = 0.025
    frame_stride = 0.01
    
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    
    frames *= numpy.hamming(frame_length)
    if frameAdmission:
        mag_frames = getAdmittedFrames(frames, NFFT, frameAdmission)
    else:
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    if len(pow_frames) <= 0:
        return None
    
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    
    return filter_banks


def getAdmittedFrames(frames, blocksize, blocklimit):
    """
    frames: 
    blocksize:
    blocklimit: 0 or more for limiting the block
                -1 for admitting every frame
    """
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, blocksize))  # Magnitude of the FFT
    admittedFrames = []
    if blocklimit >= 0:
        entropyThreshold = 2.0
        rmsThreshold = 0.005
        i = 0
        frameNotAdmittedCount = 0
        while i < len(frames):
            rms = getRMS(frames[i])
            specEntropy = getSpectralEntropy(mag_frames[i], entropyThreshold)
            if (specEntropy < entropyThreshold) | (rms > rmsThreshold):
                admittedFrames.append(mag_frames[i])
                frameNotAdmittedCount = 0
            elif (frameNotAdmittedCount <= blocklimit):
                admittedFrames.append(mag_frames[i])
                frameNotAdmittedCount += 1            
            i += 1
    else:
        admittedFrames.append(mag_frames[i])
    return np.array(admittedFrames)


def getSpectralEntropy(mag_frame, entropyThreshold):
    tot = 0 # Sum
    specEntropy = 0
    numUnfoldedBins = len(mag_frame)/2
    i = 0
    while i < numUnfoldedBins:
        tot += mag_frame[i]
        i += 1
    if tot==0:
        delta = 1
    else:
        delta = 1/tot
    j = 0
    while j < numUnfoldedBins:
        prob = mag_frame[i] * delta
        if prob < 0.0000000001:
            specEntropy = entropyThreshold + 1
        else:
            specEntropy += prob * numpy.log10(prob)
        j += 1
    return abs(specEntropy)


def getRMS(samples):
    tot = 0.
    for i, sample in enumerate(samples):
        tot += sample ** 2
    rms = math.sqrt(tot/len(samples))
    return rms


