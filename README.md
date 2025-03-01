# Detecting Arrhythmia

This project demonstrates a data analysis workflow for detecting arrhythmia using ECG signal data. The analysis is implemented in the Jupyter Notebook `Detecting_Arrhhytmia.ipynb` and uses libraries such as Pandas, Matplotlib, Seaborn, and TensorFlow.

## Overview

The notebook performs the following steps:

- **Data Loading and Inspection:**  
  Reads in the training (`arrhythmia_train.csv`) and testing (`arrhythmia_test.csv`) datasets. It checks for missing values, inspects the dataset shape, and provides summary statistics of the features and class labels.

- **Exploratory Data Analysis (EDA):**  
  - Displays the head of the datasets and their descriptions.
  - Shows the distribution of the arrhythmia classes using a bar plot with class labels such as *Normal beat (class 0)*, *Supraventricular ectopic beat (class 1)*, *Ventricular ectopic beat (class 2)*, and *Unknown beats (class 4)*.
  - Visualizes a subset of the ECG signals by plotting 10 normal heartbeats.
  - Extracts and plots specific features (e.g., comparing Supraventricular and Ventricular heartbeats).

- **Feature Visualization:**  
  A scatter plot is created using two selected time points (features) with class-dependent colors to further analyze the differences between the arrhythmia classes.

- **Deep Learning Setup:**  
  Although TensorFlow is imported for potential model building, the notebook currently focuses on data exploration and visualization. Note that messages regarding missing CUDA drivers indicate that the GPU is not used and the computations will run on the CPU.

## Data Description

- **Dataset Format:**  
  - **Records:** The training dataset contains 112,210 records, while the test dataset contains 21,729 records.
  - **Attributes:** Each record has 188 columns – 187 columns representing the ECG signal as a 1D time series (sampled at 300Hz) and 1 column for the arrhythmia class label.
  - **Classes:** The arrhythmia labels include multiple classes (e.g., 0, 1, 2, and 4), corresponding to different heartbeat types.

## Requirements

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- TensorFlow

You can install the required packages using pip:

```bash
pip install pandas matplotlib seaborn tensorflow
```

## How to Run

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/yourusername/detecting-arrhythmia.git
   cd detecting-arrhythmia
   ```

2. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook Detecting_Arrhhytmia.ipynb
   ```

3. **Run the Notebook Cells:**  
   Follow the sequential steps in the notebook to execute the data analysis and view the generated plots.

## Code Highlights

Below are some key excerpts from the analysis:

### Data Loading and Summary

```python
import pandas as pd

# Load the training and testing datasets
df_train = pd.read_csv('arrhythmia_train.csv')
df_test = pd.read_csv('arrhythmia_test.csv')

# Display basic information and descriptive statistics
print(df_train.shape)
print(df_train.describe())
print(df_train.isnull().sum())
```

### Visualizing Class Distribution

```python
import matplotlib.pyplot as plt

ax = df_train['arrhythmia'].value_counts().plot(kind='bar')
ax.set_xticklabels(['Normal beat (class 0)', 'Supraventricular ectopic beat (class 1)',
                    'Ventricular ectopic beat (class 2)', 'Unknown beats (class 4)'])
plt.title('Arrhythmia Class Distribution')
plt.show()
```

### Plotting ECG Signals

```python
# Plot 10 normal heartbeats
nprmal_beats = df_train[df_train['arrhythmia'] == 0]
plt.figure(figsize=(20, 10))
for i in range(10):
    heartbeat = nprmal_beats.iloc[i, :-1]
    plt.plot(heartbeat, label=f'Normal heartbeat {i+1}')
plt.title("10 Normal Heartbeats", fontsize=20)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("ECG Signal Amplitude", fontsize=16)
plt.legend()
plt.show()
```

### Comparing Specific Arrhythmia Features

```python
# Extract and plot features for Supraventricular and Ventricular beats
feature_1 = df_train[df_train['arrhythmia'] == 1].iloc[0, 1:-1]
feature_2 = df_train[df_train['arrhythmia'] == 2].iloc[0, 1:-1]

plt.figure(figsize=(20, 10))
plt.plot(feature_1, label='Supraventricular')
plt.plot(feature_2, label='Ventricular')
plt.title('Supraventricular and Ventricular', fontsize=20)
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('ECG Signal Amplitude', fontsize=16)
plt.legend()
plt.show()
```

### Feature Scatter Plot

```python
import seaborn as sns

# Select two features (time points) and create a scatter plot by class
feature_1 = df_train.columns[1]  
feature_2 = df_train.columns[2]  
sns.scatterplot(x=feature_1, y=feature_2, hue='arrhythmia', data=df_train)
plt.title('Feature Scatter Plot by Class')
plt.show()
```

## Conclusion

This notebook provides a comprehensive walkthrough for:
- Loading and inspecting the ECG arrhythmia dataset.
- Exploring and visualizing the signal patterns.
- Comparing different arrhythmia classes using plots.
