# Finite Impulse Response (FIR) band-pass filter for processing time-series data from brain parcellation
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew, kurtosis
from scipy.signal import welch, convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Path to your .vhdr file
#vhdr_file = 'C:\\Users\\user\\Documents\\DSP\\Data\\sub-1824\\eeg\\sub-1824_task-Rest_eeg.vhdr'

# Load the data
#raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
#print(raw.info)  # This will print out the metadata and channel information

# Plot the data to visualize EEG signals
#raw.plot()
#plt.show()

# Load EEG data
#eeg_data = pd.read_csv('C:\\Users\\user\\Documents\\DSP\\Data\\sub-1824\\eeg\\sub-1824_task-Rest_channels.tsv')
#eeg_signal = eeg_data.iloc[:, 0]  # Assuming EEG data is in the first column

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Define the sinc function to create filter coefficients
def sinc_filter(cutoff, N, fs, type='low'):
    n = np.arange(N) - (N - 1) / 2
    h = np.sinc(2 * cutoff / fs * n)
    if type == 'high':
        h = np.sinc(n) - h  # High-pass filter using spectral inversion
    return h * np.hamming(N)  # Applying a Hamming window

# Sampling and signal parameters
fs = 1024  # Sampling frequency in Hz
duration = 10.0  # Total duration of signal in seconds
num_segments = 15  # Number of segments or samples
t = np.arange(0, duration, 1/fs)
segment_length = int(len(t) / num_segments)  # Calculate segment length

# Generate synthetic EEG data
np.random.seed(42)
noise = np.random.randn(len(t)) * 5
alpha_wave = 50 * np.sin(2 * np.pi * 10 * t)  # 10 Hz wave
beta_wave = 30 * np.sin(2 * np.pi * 20 * t)   # 20 Hz wave
synthetic_signal = noise + alpha_wave + beta_wave

# Frequency band parameters
low_cutoff = 30 # Low cutoff frequency (Hz)
high_cutoff = 70 # High cutoff frequency (Hz)
numtaps = 101 # Number of filter taps (should be odd)

# Create band-pass filter using the difference of low-pass filters
h_low = sinc_filter(high_cutoff, numtaps, fs, 'low')
h_high = sinc_filter(low_cutoff, numtaps, fs, 'high')
h_bp = h_high - h_low
filtered_signal = np.convolve(synthetic_signal, h_bp, mode='same')

# Make sure to slice 't' to match the length of 'filtered_signal' for plotting
t_filtered = t[:len(filtered_signal)]

# Feature Extraction: Power Spectral Density
f, psd = welch(filtered_signal, fs=fs)
plt.figure(figsize=(6, 4))
plt.plot(f, psd)
plt.title('Power Spectral Density of Filtered Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.show()

# Prepare features (mean and standard deviation of PSD values)
# Extract more features
features = []
labels = np.random.randint(0, 2, num_segments)  # Binary labels for each segment
for i in range(num_segments):
    start = int(i * segment_length)
    end = int(start + segment_length)
    segment_signal = synthetic_signal[start:end]
    filtered_segment = convolve(segment_signal, h_bp, mode='same')
    f, psd = welch(filtered_segment, fs=fs)
    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    skewness_psd = skew(psd)
    kurtosis_psd = kurtosis(psd)
    features.append([mean_psd, std_psd, skewness_psd, kurtosis_psd])

features = np.array(features)

# Train a simple logistic regression model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# Train and evaluate different models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='linear'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Best Random Forest: {accuracy:.2f}')

# Cross-validation with Logistic Regression
model = LogisticRegression()
cv_scores = cross_val_score(model, features, labels, cv=5)
print(f'Cross-Validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')


# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(t, synthetic_signal)
plt.title('Original (Synthetic) EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(312)
plt.plot(t_filtered, filtered_signal)
plt.title('Filtered Signal (30-70 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(313)
plt.stem(np.arange(numtaps) - (numtaps - 1) / 2, h_bp, basefmt=" ", linefmt='blue', markerfmt='bo')
plt.title('Impulse Response of the Band-Pass FIR Filter')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# 1. Separate filter coefficient generation from calculation.
# 2. Modify another filter (integer-based). Ground coefficients to nearest integer, digitized filter may or may not behave similar to current filter. 
# 3. "How do we choose the result?" More or less fractional bits, rounding, etc. Creating a filter which is operating entirely on bit sequences (binary fractionals), along with what is recommended frequency. Design exploration, how many filter coefficients would we need in the first place, bitness (width).
# 4. Main point: see trade-off between parameters. Low pass, then low pass + high pass (== band pass). Consider trying band-stop further. Can we integrate number of coefficients. 
# Not a good idea to add 3 or more 32-bit numbers concurrently.  