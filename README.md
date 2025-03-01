# Detecting Arrhythmia

This project demonstrates a data analysis workflow for detecting arrhythmia using ECG signal data. The analysis is implemented in the Jupyter Notebook `Detecting_Arrhhytmia.ipynb` and uses libraries such as Pandas, Matplotlib, Seaborn, and TensorFlow.

## Overview

The notebook performs the following steps:

- **Data Loading and Inspection**  
  Reads in the training (`arrhythmia_train.csv`) and testing (`arrhythmia_test.csv`) datasets. It checks for missing values, inspects the dataset shape, and provides summary statistics of the features and class labels.

- **Exploratory Data Analysis (EDA)**  
  - Displays the head of the datasets and their descriptions.
  - Shows the distribution of the arrhythmia classes using a bar plot with class labels such as *Normal beat (class 0)*, *Supraventricular ectopic beat (class 1)*, *Ventricular ectopic beat (class 2)*, and *Unknown beats (class 4)*.
  - Visualizes subsets of the ECG signals (e.g., 10 normal heartbeats, Supraventricular vs. Ventricular heartbeats).
  - Creates a feature scatter plot (two selected time points) to observe how different arrhythmia classes cluster.

- **Insights and Observations**  
  Incorporates both visual and basic numerical insights gleaned from the plots and simple code snippets (shown below).

- **Deep Learning Setup**  
  Although TensorFlow is imported for potential model building, the notebook currently focuses on data exploration and visualization. Messages about missing CUDA drivers indicate that the GPU is not used, so computations will run on the CPU.

## Data Description

- **Dataset Format:**  
  - **Records:**  
    - Training: 112,210 records  
    - Testing: 21,729 records
  - **Attributes:**  
    Each record has 188 columns – 187 columns representing the ECG signal as a 1D time series (sampled at 300Hz) and 1 column for the arrhythmia class label.
  - **Classes:**  
    The arrhythmia labels include multiple classes (e.g., 0, 1, 2, and 4), corresponding to different heartbeat types.

## Visual Insights

### 1. Ten Normal Heartbeats

![image](https://github.com/user-attachments/assets/b8de94f3-9383-49c0-9dd8-45dd222ab785)



- **Shape & Amplitude Ranges**  
  - Each of the 10 plotted signals peaks at different times, but most amplitudes stay between 0.0 and 1.0 (indicating normalized ECG data).
  - Some heartbeats exhibit one strong peak, while others have multiple smaller fluctuations.

- **Potential Usage**  
  - Such variability suggests that while normal beats can differ, they often share certain patterns or amplitude ranges. These patterns can help distinguish them from abnormal beats in a classification model.

### 2. Supraventricular vs. Ventricular Example

![image](https://github.com/user-attachments/assets/1b6f1c6a-b1b6-4a3c-ae68-ebedf024418a)


- **Signal Differences**  
  - The *Supraventricular* example (blue line) shows a sharp rise around an early time (~ index 50-60), and then descends to a moderate amplitude.
  - The *Ventricular* example (orange line) shows a pronounced spike near the mid-to-later portion of the signal (~ index 120), after which it drops toward zero.

- **Implication**  
  - These distinct waveform shapes hint that certain time segments (peaks or drops) could be highly indicative of different arrhythmias.
  - Feature engineering (e.g., focusing on peak amplitude positions) might help a machine learning model differentiate between these arrhythmia types.

### 3. Feature Scatter Plot by Class

![image](https://github.com/user-attachments/assets/b5b40d4b-ad55-47ac-b557-68d38df892f9)



- **Scatter Distribution**  
  - This plot shows two selected features (time points `X2` and `X3`) with color-coded classes.  
  - Classes 2 and 4 tend to cluster in the top-right region, while class 0 is generally spread in the lower-left region, and class 1 is in-between.

- **Interpretation**  
  - Although there is some overlap, these two features alone already show partial separability of the classes.  
  - In practice, combining more features/time points in a model will likely yield stronger separation and better classification performance.

## Simple Code-Based Insights

To complement the visual analysis, you can run small code snippets to gather basic numerical insights, such as average amplitudes across different arrhythmia classes.

```python
import numpy as np

# Separate beats by class (0: Normal, 1: Supraventricular, 2: Ventricular)
normal_beats = df_train[df_train['arrhythmia'] == 0].iloc[:, :-1].values
supra_beats = df_train[df_train['arrhythmia'] == 1].iloc[:, :-1].values
vent_beats = df_train[df_train['arrhythmia'] == 2].iloc[:, :-1].values

print("Average amplitude (Normal):", np.mean(normal_beats))
print("Average amplitude (Supraventricular):", np.mean(supra_beats))
print("Average amplitude (Ventricular):", np.mean(vent_beats))
```

<details>
<summary>Example Interpretation of Results (Hypothetical)</summary>

```
Average amplitude (Normal): 0.40
Average amplitude (Supraventricular): 0.43
Average amplitude (Ventricular): 0.50
```

- **Insight:** Ventricular beats might have a higher average amplitude than Normal or Supraventricular beats, which aligns with the stronger spike observed in the Ventricular example plot.
</details>

You can also check peak amplitudes for a small subset of signals:

```python
# Peak amplitude for the first 10 Normal beats
peak_amplitudes = []
for i in range(10):
    row = normal_beats[i]
    peak_amplitudes.append(row.max())

print("Peak amplitudes for the first 10 Normal beats:", peak_amplitudes)
print("Average peak amplitude for these 10 Normal beats:", np.mean(peak_amplitudes))
```

Such quick calculations can reinforce the visual observations (e.g., normal beats often peak around ~0.8 to ~1.0 in the plotted examples).

## Requirements

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- TensorFlow

Install these using:

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

## Conclusion

- The plots reveal that **Normal** heartbeats have moderate variability but generally share characteristic amplitude ranges.
- **Supraventricular** vs. **Ventricular** signals can differ significantly in the timing and height of their main peaks.
- Even with just two features (`X2` vs. `X3`), the scatter plot shows partial separation of classes, suggesting that more features/time points could enhance arrhythmia classification.
