# adopted from https://github.com/ruizhecao96/CMGAN/blob/main/src/tools/compute_metrics.py

import numpy as np
from scipy.io import wavfile
from scipy.linalg import toeplitz, norm
from scipy.fftpack import fft
import math
from scipy import signal
from pesq import pesq
from numba import njit

""" 
This is a python script which can be regarded as implementation of matlab script "compute_metrics.m".

Usage: 
    pesq, csig, cbak, covl, ssnr, stoi = compute_metrics(cleanFile, enhancedFile, Fs, path)
    cleanFile: clean audio as array or path if path is equal to 1
    enhancedFile: enhanced audio as array or path if path is equal to 1
    Fs: sampling rate, usually equals to 8000 or 16000 Hz
    path: whether the "cleanFile" and "enhancedFile" arguments are in .wav format or in numpy array format, 
          1 indicates "in .wav format"
          
Example call:
    pesq_output, csig_output, cbak_output, covl_output, ssnr_output, stoi_output = \
            compute_metrics(target_audio, output_audio, 16000, 0)
"""


def compute_metrics(cleanFile, enhancedFile, Fs, path):
    alpha = 0.95

    if path == 1:
        sampling_rate1, data1 = wavfile.read(cleanFile)
        sampling_rate2, data2 = wavfile.read(enhancedFile)
        if sampling_rate1 != sampling_rate2:
            raise ValueError("The two files do not match!\n")
    else:
        data1 = cleanFile
        data2 = enhancedFile
        sampling_rate1 = Fs
        sampling_rate2 = Fs

    if len(data1) != len(data2):
        length = min(len(data1), len(data2))
        data1 = data1[0:length] + np.spacing(1)
        data2 = data2[0:length] + np.spacing(1)

    # compute the WSS measure
    wss_dist_vec = wss(data1, data2, sampling_rate1)
    wss_dist_vec = np.sort(wss_dist_vec)
    wss_dist = np.mean(wss_dist_vec[0 : round(np.size(wss_dist_vec) * alpha)])

    # compute the LLR measure
    LLR_dist = llr(data1, data2, sampling_rate1)
    LLRs = np.sort(LLR_dist)
    LLR_len = round(np.size(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[0:LLR_len])

    # compute the SNRseg
    snr_dist, segsnr_dist = snr(data1, data2, sampling_rate1)
    snr_mean = snr_dist
    segSNR = np.mean(segsnr_dist)

    # compute the pesq
    pesq_mos = pesq(sampling_rate1, data1, data2, "wb")

    # now compute the composite measures
    CSIG = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    CSIG = max(1, CSIG)
    CSIG = min(5, CSIG)  # limit values to [1, 5]
    CBAK = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    CBAK = max(1, CBAK)
    CBAK = min(5, CBAK)  # limit values to [1, 5]
    COVL = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    COVL = max(1, COVL)
    COVL = min(5, COVL)  # limit values to [1, 5]

    STOI = stoi(data1, data2, sampling_rate1)

    return pesq_mos, CSIG, CBAK, COVL, segSNR, STOI


import numpy as np
from scipy.fftpack import fft

def wss(clean_speech, processed_speech, sample_rate):
    # Ensure the signals have the same length
    if len(clean_speech) != len(processed_speech):
        raise ValueError("Files must have the same length.")
    
    # Global variables
    winlength = int(round(30 * sample_rate / 1000))  # window length in samples
    skiprate = winlength // 4  # window skip in samples
    max_freq = sample_rate / 2  # maximum bandwidth
    num_crit = 25  # number of critical bands

    n_fft = int(2 ** np.ceil(np.log2(2 * winlength)))
    n_fftby2 = n_fft // 2  # FFT size/2
    Kmax = 20.0  # value suggested by Klatt, pg 1280
    Klocmax = 1.0  # value suggested by Klatt, pg 1280

    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    cent_freq = np.array([
        50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000,
        540.000, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30,
        1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71,
        2701.97, 2978.04, 3276.17, 3597.63,
    ])
    bandwidth = np.array([
        70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000,
        77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423,
        153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255,
        276.072, 298.126, 321.465, 346.136,
    ])

    bw_min = bandwidth[0]  # minimum critical bandwidth

    # Set up the critical band filters
    min_factor = np.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter
    j = np.arange(n_fftby2)
    crit_filter = np.zeros((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        crit_filter[i, :] = np.exp(-11 * ((j - np.floor(f0)) / bw) ** 2 + norm_factor)
        crit_filter[i, crit_filter[i, :] < min_factor] = 0.0

    # Calculate the number of frames
    num_frames = int(len(clean_speech) / skiprate - (winlength / skiprate))

    # Use the same window as in the original code
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    distortion = np.zeros(num_frames)
    start = 0
    for frame_count in range(num_frames):
        # Get frames and apply window
        clean_frame = clean_speech[start:start + winlength] / 32768.0
        processed_frame = processed_speech[start:start + winlength] / 32768.0
        clean_frame *= window
        processed_frame *= window

        # Compute the Power Spectrum
        clean_spec = np.abs(fft(clean_frame, n_fft)) ** 2
        processed_spec = np.abs(fft(processed_frame, n_fft)) ** 2

        # Compute Filterbank Output Energies (in dB scale)
        clean_energy = np.dot(crit_filter, clean_spec[:n_fftby2])
        processed_energy = np.dot(crit_filter, processed_spec[:n_fftby2])

        clean_energy = 10 * np.log10(np.maximum(clean_energy, 1e-10))
        processed_energy = 10 * np.log10(np.maximum(processed_energy, 1e-10))

        # Compute Spectral Slope
        clean_slope = clean_energy[1:] - clean_energy[:-1]
        processed_slope = processed_energy[1:] - processed_energy[:-1]

        # Find the nearest peak locations in the spectra to each critical band
        clean_loc_peak = np.zeros(num_crit - 1)
        processed_loc_peak = np.zeros(num_crit - 1)

        for i in range(num_crit - 1):
            # Clean speech peaks
            n = i
            if clean_slope[i] > 0:
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak[i] = clean_energy[n - 1]
            else:
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak[i] = clean_energy[n + 1]

            # Processed speech peaks
            n = i
            if processed_slope[i] > 0:
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak[i] = processed_energy[n - 1]
            else:
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak[i] = processed_energy[n + 1]

        # Compute the WSS Measure for this frame
        dBMax_clean = np.max(clean_energy)
        dBMax_processed = np.max(processed_energy)

        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:-1])
        W_clean = Wmax_clean * Wlocmax_clean

        Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:-1])
        W_processed = Wmax_processed * Wlocmax_processed

        W = (W_clean + W_processed) / 2.0
        slope_diff = clean_slope - processed_slope
        distortion[frame_count] = np.sum(W * slope_diff ** 2) / np.sum(W)

        start += skiprate

    return distortion



def llr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech.  Must be the same.
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    # Global Variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(
        int
    )  # window length in samples
    skiprate = (np.floor(winlength / 4)).astype(int)  # window skip in samples
    if sample_rate < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int((clean_length - winlength) / skiprate)  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Get the autocorrelation lags and LPC parameters used to compute the LLR measure.
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)

        # (3) Compute the LLR measure
        numerator = np.dot(np.matmul(A_processed, toeplitz(R_clean)), A_processed)
        denominator = np.dot(np.matmul(A_clean, toeplitz(R_clean)), A_clean)
        distortion[frame_count] = math.log(numerator / denominator)
        start = start + skiprate
    return distortion


def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocorrelation Lags
    winlength = np.size(speech_frame)
    R = np.empty(model_order + 1)
    E = np.empty(model_order + 1)
    for k in range(model_order + 1):
        R[k] = np.dot(speech_frame[0 : winlength - k], speech_frame[k:winlength])

    # (2) Levinson-Durbin
    a = np.ones(model_order)
    a_past = np.empty(model_order)
    rcoeff = np.empty(model_order)
    E[0] = R[0]
    for i in range(model_order):
        a_past[0:i] = a[0:i]
        sum_term = np.dot(a_past[0:i], R[i:0:-1])
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i == 0:
            a[0:i] = a_past[0:i] - np.multiply(a_past[i - 1 : -1 : -1], rcoeff[i])
        else:
            a[0:i] = a_past[0:i] - np.multiply(a_past[i - 1 :: -1], rcoeff[i])
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = R
    refcoeff = rcoeff
    lpparams = np.concatenate((np.array([1]), -a))
    return acorr, refcoeff, lpparams


def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    overall_snr = 10 * np.log10(
        np.sum(np.square(clean_speech))
        / np.sum(np.square(clean_speech - processed_speech))
    )

    # Global Variables
    winlength = round(30 * sample_rate / 1000)  # window length in samples
    skiprate = math.floor(winlength / 4)  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(
            signal_energy / (noise_energy + EPS) + EPS
        )
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr


def stoi(x: np.ndarray, y: np.ndarray, fs_signal: int) -> float:
    """
    Computes the Short-Time Objective Intelligibility (STOI) measure between two signals.

    Parameters:
    x (np.ndarray): Reference (clean) signal.
    y (np.ndarray): Test (processed) signal.
    fs_signal (int): Sampling rate of the input signals.

    Returns:
    float: STOI intelligibility score.
    """
    if x.size != y.size:
        raise ValueError("x and y should have the same length")

    # Initialization
    fs = 10000  # Sample rate for intelligibility measure
    N_frame = 256  # Window size
    K = 128  # Overlap size (changed from 512 to 128 to match N_frame / 2)
    N_fft = 512  # FFT size
    J = 15  # Number of 1/3 octave bands
    mn = 150  # Center frequency of first 1/3 octave band in Hz
    N = 30  # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15  # Lower SDR-bound
    dyn_range = 40  # Speech dynamic range

    # Obtain 1/3 octave band matrix
    H, _ = thirdoct(fs, N_fft, J, mn)  # Ensure thirdoct is correctly implemented

    # Resample signals if different sample rate is used than fs
    if fs_signal != fs:
        # Calculate the up and down factors for resampling
        # Using gcd to simplify resampling factors
        from math import gcd

        up = fs
        down = fs_signal
        factor = gcd(up, down)
        up_factor = up // factor
        down_factor = down // factor

        x = signal.resample_poly(x, up_factor, down_factor)
        y = signal.resample_poly(y, up_factor, down_factor)

    # Remove silent frames
    # Assuming removeSilentFrames is a defined function that removes frames below a certain dynamic range
    x, y = removeSilentFrames(x, y, dyn_range, N_frame, K)

    # Apply 1/3 octave band TF-decomposition using optimized stdft
    x_hat = stdft(x, N_frame, K, N_fft)  # apply short-time DFT to clean speech
    y_hat = stdft(y, N_frame, K, N_fft)  # apply short-time DFT to processed speech

    # Take single-sided spectrum
    x_hat = np.transpose(x_hat[:, : (N_fft // 2) + 1])
    y_hat = np.transpose(y_hat[:, : (N_fft // 2) + 1])

    # Apply 1/3 octave bands
    X = np.sqrt(np.matmul(H, np.square(np.abs(x_hat))))
    Y = np.sqrt(np.matmul(H, np.square(np.abs(y_hat))))

    # Initialize intermediate intelligibility measure
    num_segments = x_hat.shape[1] - N + 1
    d_interm = np.zeros(num_segments)
    c = 10 ** (-Beta / 20)  # Constant for clipping procedure

    # Precompute sums for efficiency
    X_squared = np.square(X)
    Y_squared = np.square(Y)

    # To prevent division by zero, add a small epsilon
    epsilon = 1e-10

    for m in range(num_segments):
        # Extract segments
        X_seg = X[:, m : m + N]
        Y_seg = Y[:, m : m + N]

        # Compute scale factor alpha
        numerator = np.sum(X_squared[:, m : m + N], axis=1, keepdims=True)
        denominator = np.sum(Y_squared[:, m : m + N], axis=1, keepdims=True) + epsilon
        alpha = np.sqrt(numerator / denominator)

        # Apply scaling
        aY_seg = Y_seg * alpha

        # Apply clipping
        Y_prime = np.minimum(aY_seg, X_seg + (X_seg * c))

        # Compute correlation coefficient
        d_interm[m] = taa_corr(X_seg, Y_prime) / J

    # Compute the final STOI score as the mean of intermediate measures
    d = d_interm.mean()
    return d



def thirdoct(fs, N_fft, numBands, mn):
    """
    [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
    inputs:
        FS:         samplerate
        N_FFT:      FFT size
        NUMBANDS:   number of bands
        MN:         center frequency of first 1/3 octave band
    outputs:
        A:          octave band matrix
        CF:         center frequencies
    """
    f = np.linspace(0, fs, N_fft + 1)
    f = f[0 : int(N_fft / 2 + 1)]
    k = np.arange(numBands)
    cf = np.multiply(np.power(2, k / 3), mn)
    fl = np.sqrt(
        np.multiply(
            np.multiply(np.power(2, k / 3), mn),
            np.multiply(np.power(2, (k - 1) / 3), mn),
        )
    )
    fr = np.sqrt(
        np.multiply(
            np.multiply(np.power(2, k / 3), mn),
            np.multiply(np.power(2, (k + 1) / 3), mn),
        )
    )
    A = np.zeros((numBands, len(f)))

    for i in range(np.size(cf)):
        b = np.argmin((f - fl[i]) ** 2)
        fl[i] = f[b]
        fl_ii = b

        b = np.argmin((f - fr[i]) ** 2)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii:fr_ii] = 1

    rnk = np.sum(A, axis=1)
    end = np.size(rnk)
    rnk_back = rnk[1:end]
    rnk_before = rnk[0 : (end - 1)]
    for i in range(np.size(rnk_back)):
        if (rnk_back[i] >= rnk_before[i]) and (rnk_back[i] != 0):
            result = i
    numBands = result + 2
    A = A[0:numBands, :]
    cf = cf[0:numBands]
    return A, cf


def stdft(x, N, K, N_fft):
    """
    X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time hanning-windowed dft of X with frame-size N,
    overlap K and DFT size N_FFT. The columns and rows of X_STDFT denote the frame-index and dft-bin index,
    respectively.
    """
    frames_size = int((np.size(x) - N) / K)
    w = signal.windows.hann(N + 2)
    w = w[1 : N + 1]

    x_stdft = signal.stft(
        x,
        window=w,
        nperseg=N,
        noverlap=K,
        nfft=N_fft,
        return_onesided=False,
        boundary=None,
    )[2]
    x_stdft = np.transpose(x_stdft)[0:frames_size, :]

    return x_stdft


def removeSilentFrames(x, y, dyrange, N, K):
    """
    [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y are segmented with frame-length N
    and overlap K, where the maximum energy of all frames of X is determined, say X_MAX.
    X_SIL and Y_SIL are the reconstructed signals, excluding the frames, where the energy of a frame
    of X is smaller than X_MAX-RANGE
    """

    frames = np.arange(0, (np.size(x) - N), K)
    w = signal.windows.hann(N + 2)
    w = w[1 : N + 1]

    jj_list = np.empty((np.size(frames), N), dtype=int)
    for j in range(np.size(frames)):
        jj_list[j, :] = np.arange(frames[j] - 1, frames[j] + N - 1)

    msk = 20 * np.log10(np.divide(norm(np.multiply(x[jj_list], w), axis=1), np.sqrt(N)))

    msk = (msk - np.max(msk) + dyrange) > 0
    count = 0

    x_sil = np.zeros(np.size(x))
    y_sil = np.zeros(np.size(y))

    for j in range(np.size(frames)):
        if msk[j]:
            jj_i = np.arange(frames[j], frames[j] + N)
            jj_o = np.arange(frames[count], frames[count] + N)
            x_sil[jj_o] = x_sil[jj_o] + np.multiply(x[jj_i], w)
            y_sil[jj_o] = y_sil[jj_o] + np.multiply(y[jj_i], w)
            count = count + 1

    x_sil = x_sil[0 : jj_o[-1] + 1]
    y_sil = y_sil[0 : jj_o[-1] + 1]
    return x_sil, y_sil




@njit(cache=True)
def taa_corr(x, y):
    """
    RHO = TAA_CORR(X, Y) Returns correlation coefficient between column
    vectors x and y. Optimized for performance without using 'keepdims'.
    """
    # Compute the mean for each row
    x_mean = np.sum(x, axis=1) / x.shape[1]
    y_mean = np.sum(y, axis=1) / y.shape[1]

    # Subtract mean to center data (reshaping to match dimensions for broadcasting)
    xn = x - x_mean[:, None]
    yn = y - y_mean[:, None]

    # Compute norms manually to avoid np.linalg overhead
    xn_norm = np.sqrt(np.sum(xn ** 2, axis=1) + 1e-8)
    yn_norm = np.sqrt(np.sum(yn ** 2, axis=1) + 1e-8)

    # Normalize vectors (reshaping norms to match dimensions for broadcasting)
    xn = xn / xn_norm[:, None]
    yn = yn / yn_norm[:, None]

    # Compute the correlation using trace of dot product (equivalent to sum of element-wise product)
    rho = np.sum(xn * yn)  # Equivalent to trace(xn @ yn.T) in this context

    return rho
