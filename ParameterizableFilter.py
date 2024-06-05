import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellip, lfilter, freqz
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def apply_filter():
    try:
        # Read values from GUI
        samprate = int(sample_rate_entry.get())
        lowband = int(low_band_entry.get())
        highband = int(high_band_entry.get())
        order = int(order_entry.get())
        rp = float(rp_entry.get())
        rs = float(rs_entry.get())
        rf_freq = float(rf_freq_entry.get())
        nco_freq = float(nco_freq_entry.get())
        output_width = 16  # Assuming a 16-bit width for the signal

        # Number of samples
        samples = samprate

        # Generate the RF and NCO signals
        t = np.arange(samples) / samprate
        nco_signal = np.round(np.exp(1j * (t) * (2 * np.pi) * nco_freq) * 2 ** (output_width - 1))
        rf_signal = np.round(np.exp(1j * (t) * (2 * np.pi) * rf_freq) * 2 ** (output_width - 1))

        # Generate the signals constrained to fixed sampling
        ncoAReal = np.round(np.sin((t) * (2 * np.pi) * nco_freq) * 2 ** (output_width - 1))
        ncoA = np.zeros(samples, complex)
        ncoA.imag = ncoAReal
        ncoA.real = np.round(np.cos((t) * (2 * np.pi) * nco_freq) * 2 ** (output_width - 1))

        rfInADC = np.round(np.sin((t) * (2 * np.pi) * rf_freq) * 2 ** (output_width - 1))
        rfInComplex = np.zeros(samples, complex)
        rfInComplex.imag = rfInADC
        rfInComplex.real = np.round(np.cos((t) * (2 * np.pi) * rf_freq) * 2 ** (output_width - 1))

        f0 = (np.conj(nco_signal) * rf_signal)  # ideal
        f1 = (np.conj(ncoA) * rfInComplex)  # FPGA constrained to fixed sampling
        f0.imag = f0.imag / 2 ** 16
        f0.real = f0.real / 2 ** 16

        # Mix RF with NCO
        mixed_signal = rf_signal * np.conj(nco_signal)

        # Design the elliptic band-pass filter
        b, a = ellip(order, rp, rs, [lowband, highband], btype='bandpass', fs=samprate)

        # Apply the filter
        filtered_signal = lfilter(b, a, mixed_signal)

        # Compute the frequency response of the filter
        w, h = freqz(b, a, worN=8000, fs=samprate)

        # Compute FFT of the signals
        fft_rf = np.fft.fft(rf_signal)
        fft_filtered = np.fft.fft(filtered_signal)
        freqs = np.fft.fftfreq(len(t), 1/samprate)

        # Clear previous plots
        for ax in axs.flat:
            ax.clear()

        # Plot the filter frequency response
        axs[0, 0].plot(w, 20 * np.log10(np.abs(h)), label='Filter Response')
        axs[0, 0].set_title('Filter Frequency Response')
        axs[0, 0].set_xlabel('Frequency [Hz]')
        axs[0, 0].set_ylabel('Gain [dB]')
        axs[0, 0].grid(True)

        # Plot the FFT of the original RF signal
        axs[0, 1].plot(freqs[:samples//2], 20 * np.log10(np.abs(fft_rf)[:samples//2]), label='Original RF')
        axs[0, 1].set_xlim([0, samprate/2])
        axs[0, 1].set_title('FFT of Original RF Signal')
        axs[0, 1].set_xlabel('Frequency [Hz]')
        axs[0, 1].set_ylabel('Magnitude [dB]')
        axs[0, 1].grid(True)

        # Plot the FFT of the filtered signal
        axs[1, 0].plot(freqs[:samples//2], 20 * np.log10(np.abs(fft_filtered)[:samples//2]), label='Filtered Signal', color='red')
        axs[1, 0].set_xlim([0, samprate/2])
        axs[1, 0].set_title('FFT of Filtered Signal')
        axs[1, 0].set_xlabel('Frequency [Hz]')
        axs[1, 0].set_ylabel('Magnitude [dB]')
        axs[1, 0].grid(True)

        # Plot the time-domain signals
        axs[1, 1].plot(t, np.real(rf_signal), label='Original RF')
        axs[1, 1].plot(t, np.real(filtered_signal), label='Filtered Signal', color='red')
        axs[1, 1].set_title('Time-Domain Signals')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].set_ylabel('Amplitude')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        canvas.draw()
    except Exception as e:
        print("Error:", e)

# Create the main window
root = Tk()
root.title("Band-Pass Filter Configuration")
root.geometry("1200x800")  # Set the window size to fit everything

# Use notebook from ttk for tabbed interface
notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Filter Settings')
notebook.pack(expand=True, fill='both')

# Entries and labels frame
entry_frame = ttk.Frame(tab1)
entry_frame.pack(side=TOP, fill=X, padx=10, pady=10)

# Entries
labels = ['Sample Rate', 'RF Freq', 'NCO Freq', 'Low Band Freq', 'High Band Freq', 'Order', 'Passband Ripple', 'Stopband Attenuation']
entries = []
for i, label_text in enumerate(labels):
    label = ttk.Label(entry_frame, text=label_text)
    label.grid(row=i, column=0, padx=5, pady=5, sticky=W)
    entry = ttk.Entry(entry_frame)
    entry.grid(row=i, column=1, padx=5, pady=5, sticky=W)
    entries.append(entry)

# Set default values for entries
defaults = ['1000000', '100000', '100000', '90000', '110000', '8', '1', '40']
for entry, default in zip(entries, defaults):
    entry.insert(0, default)

# Save references to entries globally or pass to `apply_filter`
sample_rate_entry, rf_freq_entry, nco_freq_entry, low_band_entry, high_band_entry, order_entry, rp_entry, rs_entry = entries

# Button to apply filter settings
apply_button = ttk.Button(entry_frame, text='Apply Filter', command=apply_filter)
apply_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Create a matplotlib figure and embed it in the Tkinter window
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
canvas = FigureCanvasTkAgg(fig, master=tab1)  # Embedding in the first tab
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=TOP, fill=BOTH, expand=True)

root.mainloop()


