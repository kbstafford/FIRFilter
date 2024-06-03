# README

## Finite Impulse Response (FIR) Band-Pass Filter for EEG Data Processing

This code demonstrates the implementation of a Finite Impulse Response (FIR) band-pass filter for processing time-series data obtained from brain parcellation. It utilizes various Python libraries for signal processing, machine learning, and data visualization.

### Requirements
- Python 3.x
- Required libraries: `numpy`, `matplotlib`, `mne`, `scipy`, `sklearn`

### Instructions

1. **Data Loading**: 
   - Ensure that you have EEG data in the BrainVision format (`.vhdr` file).
   - Uncomment and modify the `vhdr_file` variable to specify the path to your `.vhdr` file.
   - Uncomment the relevant code to load the EEG data using `mne.io.read_raw_brainvision()`.

2. **Data Visualization**:
   - Visualize the EEG signals using `raw.plot()` after loading the data.

3. **Synthetic EEG Data Generation**:
   - Synthetic EEG data is generated for demonstration purposes.
   - You can adjust parameters such as sampling frequency, duration, and frequency components of the synthetic signal.

4. **Filter Design**:
   - Define the sinc function to create filter coefficients using `sinc_filter()`.
   - Specify the desired frequency band using `low_cutoff` and `high_cutoff`.
   - Number of filter taps (`numtaps`) can be adjusted for desired filter characteristics.

5. **Signal Filtering**:
   - Apply the band-pass filter to the synthetic EEG signal using `np.convolve()`.

6. **Feature Extraction**:
   - Extract features from the filtered signal, such as mean, standard deviation, skewness, and kurtosis of the Power Spectral Density (PSD) values.

7. **Model Training and Evaluation**:
   - Train and evaluate machine learning models (Logistic Regression, Random Forest, SVM, Neural Network) using extracted features.
   - Perform hyperparameter tuning for the Random Forest classifier.
   - Evaluate model performance using accuracy metrics and cross-validation.

8. **Plotting**:
   - Visualize the original synthetic EEG signal, the filtered signal, and the impulse response of the band-pass FIR filter.

### Usage
- Ensure that all required libraries are installed (`numpy`, `matplotlib`, `mne`, `scipy`, `sklearn`).
- Modify the code as necessary to adapt it to your specific EEG data and analysis requirements.
- Execute the code in a Python environment.

### Note
- This code serves as a demonstration and may require modifications for use with actual EEG data.
- Ensure proper understanding of the algorithms and methods employed before applying them to real-world data.
