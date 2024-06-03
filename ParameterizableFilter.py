import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellip, lfilter
import math
from tkinter import *
from tkinter import ttk  # Improved styling
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def apply_filter():
    try:
        # Read values from GUI
        samprate = int(sample_rate_entry.get())
        lowband = int(low_band_entry.get())
        highband = int(high_band_entry.get())
        numtaps = int(num_taps_entry.get())
        order = int(order_entry.get())
        rp = float(rp_entry.get())
        rs = float(rs_entry.get())

        # Center frequency for complex signal
        center_freq = math.sqrt(lowband * highband)
        t = np.linspace(0, 1, int(samprate), endpoint=False)
        complex_signal = np.exp(1j * 2 * np.pi * center_freq * t)

        # Design the elliptic band-pass filter
        b, a = ellip(order, rp, rs, [lowband, highband], btype='bandpass', fs=samprate)

        # Apply the filter to the complex signal
        filtered_signal = lfilter(b, a, complex_signal)

        # Clear previous plots
        for ax in axs.flat:
            ax.clear()

        # Plotting the complex signal and filtered signal
        plot_signals(t, complex_signal, filtered_signal)
        canvas.draw()
    except Exception as e:
        print("Error:", e)

def plot_signals(t, complex_signal, filtered_signal):
    # Subplot for Real vs. Imaginary of the original signal
    axs[0, 0].scatter(complex_signal.real, complex_signal.imag, color='blue', marker='o', label='Original')
    axs[0, 0].scatter(filtered_signal.real, filtered_signal.imag, color='red', marker='x', label='Filtered')
    axs[0, 0].set_title('Complex Plane')
    axs[0, 0].set_xlabel('Real')
    axs[0, 0].set_ylabel('Imaginary')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')

    axs[0, 1].plot(t, np.abs(complex_signal), label='Magnitude Original', color='blue')
    axs[0, 1].plot(t, np.abs(filtered_signal), label='Magnitude Filtered', color='red')
    axs[0, 1].set_title('Magnitude Over Time')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Magnitude')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(t, np.angle(complex_signal), label='Phase Original', color='blue')
    axs[1, 0].plot(t, np.angle(filtered_signal), label='Phase Filtered', color='red')
    axs[1, 0].set_title('Phase Over Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Phase (radians)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(t, complex_signal.real, label='Real Part Original', color='blue')
    axs[1, 1].plot(t, filtered_signal.real, label='Real Part Filtered', color='red')
    axs[1, 1].plot(t, complex_signal.imag, label='Imaginary Part Original', linestyle='--', color='blue')
    axs[1, 1].plot(t, filtered_signal.imag, label='Imaginary Part Filtered', linestyle='--', color='red')
    axs[1, 1].set_title('Real and Imaginary Parts Over Time')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

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
labels = ['Sample Rate', 'RF Freq', 'NCO Freq', 'Low Band Freq', 'High Band Freq', 'Number of Coefficients', 'Order', 'Passband Ripple', 'Stopband Attenuation']
entries = []
for i, label_text in enumerate(labels):
    label = ttk.Label(entry_frame, text=label_text)
    label.grid(row=i, column=0, padx=5, pady=5, sticky=W)
    entry = ttk.Entry(entry_frame)
    entry.grid(row=i, column=1, padx=5, pady=5, sticky=W)
    entries.append(entry)

# Set default values for entries
defaults = ['100000', '50000', '1000', '1', '2', '101', '8', '1', '40']
for entry, default in zip(entries, defaults):
    entry.insert(0, default)

# Save references to entries globally or pass to `apply_filter`
sample_rate_entry, rf_freq_entry, nco_freq_entry, low_band_entry, high_band_entry, num_taps_entry, order_entry, rp_entry, rs_entry = entries

# Button to apply filter settings
apply_button = ttk.Button(entry_frame, text='Apply Filter', command=apply_filter)
apply_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Create a matplotlib figure and embed it in the Tkinter window
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
canvas = FigureCanvasTkAgg(fig, master=tab1)  # Embedding in the first tab
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=TOP, fill=BOTH, expand=True)

root.mainloop()
