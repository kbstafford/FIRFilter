# Import required dependencies
import numpy as np
import matplotlib.pyplot as plt



# Filter requirements
fs = 100  # Sample rate
num_samples = 1000  # Number of samples
t = np.arange(num_samples) / fs  # Time array
cutoff_high = 80  # High-frequency cutoff
cutoff_low = 30  # Low-frequency cutoff
num_taps = 1000  # Number of filter coefficients
cutoff_freq = 2 * min(cutoff_high, cutoff_low) / fs  # Normalize cutoff frequency

# Generate complex input signal
f1 = 5  # Frequency of first component (Hz)
f2 = 15  # Frequency of second component (Hz)
f3 = 25  # Frequency of third component (Hz)
input_signal_real = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
input_signal_imag = 0.5 * np.cos(2 * np.pi * f1 * t) + 0.3 * np.cos(2 * np.pi * f2 * t) + 0.2 * np.cos(2 * np.pi * f3 * t)
input_signal = input_signal_real + 1j * input_signal_imag

# Functions for FIR filter
def coeff_gen(x, cutoff):
    if x == 0:
        return 1
    else:
        return np.sin(np.pi * cutoff * x) / (np.pi * cutoff * x)

def generate_sinc_values(cutoff, num_taps):
    n = np.arange(-(num_taps // 2), num_taps // 2 + 1)
    sinc_values = np.array([coeff_gen(x, cutoff) for x in n])
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
    output_signal = np.zeros(num_samples, dtype=complex)
    for i in range(num_samples):
        for j in range(num_taps):
            if i - j >= 0:
                output_signal[i] += input_signal[i - j] * filter_coeffs[j]
    return output_signal

# Generate Sinc Values
sinc_values = generate_sinc_values(cutoff_freq, num_taps)
print("Sinc Values:\n", sinc_values)

# Apply Windowing Function
sinc_values_windowed = apply_window(sinc_values, window='hamming')
print("Windowed Sinc Values:\n", sinc_values_windowed)

# Compute Filter Coefficients
filter_coeffs = coeff_compute(sinc_values_windowed, normalize=True)
print("Normalized Filter Coefficients:\n", filter_coeffs)

# Check if filter coefficients are normalized
coeff_sum = np.sum(filter_coeffs)
if np.isclose(coeff_sum, 1):
    print("Filter coefficients are normalized.")
else:
    print("Filter coefficients are not normalized. Sum:", coeff_sum)

# Digitize Filter Coefficients and Input Signal
filter_coeffs_digitized = coeff_digitize(filter_coeffs)
input_signal_digitized = coeff_digitize(input_signal)
print("Digitized Filter Coefficients:\n", filter_coeffs_digitized)
print("Digitized Input Signal:\n", input_signal_digitized)

# Apply FIR filter to input signal
filtered_signal = apply_fir_filter(input_signal, filter_coeffs)
filtered_signal_digitized = apply_fir_filter(input_signal_digitized, filter_coeffs_digitized)

# Plot input signal
plt.figure(figsize=(10, 5))
plt.plot(t, input_signal.real, label='Input Signal (Real)')
plt.plot(t, input_signal.imag, label='Input Signal (Imag)', linestyle='dashed')
plt.title('Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot input signal (digitized)
plt.figure(figsize=(10, 5))
plt.plot(t, input_signal_digitized.real, label='Digitized Input Signal (Real)')
plt.plot(t, input_signal_digitized.imag, label='Digitized Input Signal (Imag)', linestyle='dashed')
plt.title('Digitized Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot filtered output signal
plt.figure(figsize=(10, 5))
plt.plot(t, filtered_signal.real, label='Filtered Signal (Real)')
plt.plot(t, filtered_signal.imag, label='Filtered Signal (Imag)', linestyle='dashed')
plt.title('Filtered Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot filtered output signal (digitized)
plt.figure(figsize=(10, 5))
plt.plot(t, filtered_signal_digitized.real, label='Filtered Signal (Digitized, Real)')
plt.plot(t, filtered_signal_digitized.imag, label='Filtered Signal (Digitized, Imag)', linestyle='dashed')
plt.title('Filtered Output Signal (Digitized)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()