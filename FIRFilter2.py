# Import required dependencies
import numpy as np


# Filter requirements
fs = 100 # Sample rate
cutoff_high = 100 # High frequency cutoff
cutoff_low = 60 # Low frequency cutoff
num_taps = 51  # Number of filter coefficients
cutoff_freq = 2 * min(cutoff_high, cutoff_low) / fs  # Normalize cutoff frequency


# Functions
def coeff_gen(x, cutoff):
     if x == 0:
         return 1
     else:
         return np.sin(np.pi * cutoff * x) / (np.pi * cutoff * x)

def generate_sinc_values(cutoff, num_taps):
    sinc_values = np.array([coeff_gen(x, cutoff) for x in range(num_taps)])
    return sinc_values

def apply_window(sinc_values, window='hamming'):
    if window == 'hamming':
        window_func = np.hamming(len(sinc_values))
    elif window == 'blackman':
        window_func = np.blackman(len(sinc_values))
    # Add other windowing functions as needed
    return sinc_values * window_func

def coeff_compute(sinc_values, normalize=True):
    if normalize:
        sinc_values /= np.sum(sinc_values)
    return sinc_values

def coeff_digitize(filter_coeffs):
    return np.round(filter_coeffs)

def apply_fir_filter(input_signal, filter_coeffs):
    num_taps = len(filter_coeffs)
    num_samples = len(input_signal)
    output_signal = np.zeros(num_samples)
    for i in range(num_samples):
        for j in range(num_taps):
            if i - j >= 0:
                output_signal[i] += input_signal[i - j] * filter_coeffs[j]
    return output_signal


# Step 1: Generate Sinc Values
sinc_values = generate_sinc_values(cutoff_freq, num_taps)

# Step 2: Apply Windowing Function
sinc_values_windowed = apply_window(sinc_values, window='hamming')

# Step 3: Compute Filter Coefficients
filter_coeffs = coeff_compute(sinc_values_windowed, normalize=True)

# Example input signal (replace with your actual input signal)
input_signal = np.random.rand(1000)

# Step 4: Apply FIR filter to input signal
filtered_signal = apply_fir_filter(input_signal, filter_coeffs)

