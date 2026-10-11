\# Lab 01 Report: Environment and System Measurements



\## 1. Goal

The goal of this lab is to set up a clean Python environment, train two simple machine learning models on a breast cancer dataset, and test how much computer power they use. We want to check training time, prediction speed, memory usage, and file size to see which devices (Cloud, Edge, Mobile, or TinyML chips) can actually run them.



\## 2. Method

\- Created a separate Python 3.11 virtual environment (.venv) and installed the required libraries.

\- Created `print\_versions.py` and saved all library versions into `results/versions.txt`.

\- Loaded the Breast Cancer Wisconsin dataset and split it into 70% training data and 30% testing data with random seed 42.

\- Trained two models: Logistic Regression and a Random Forest (100 trees).

\- Saved their test accuracy into `results/baseline\_accuracy.csv`.

\- Used `measure.py` to test system performance:

&#x20; - Training time: tested 5 times after 1 warm-up run, recorded the median.

&#x20; - Inference time: tested prediction speed for one single patient 100 times, recorded the median.

&#x20; - Memory: tracked memory usage using `psutil`.

&#x20; - Model size: saved the models to disk with `joblib.dump` and measured file size in bytes and KB.



\## 3. Results



\### Baseline Accuracy

| Model | Test Accuracy |

| :--- | :--- |

| Logistic Regression | 0.9415 |

| Random Forest Classifier | 0.9357 |



\### System Measurements

| Metric | Logistic Regression | Random Forest |

| :--- | :--- | :--- |

| Median Training Time | 839.11 ms | 358.27 ms |

| Median Inference Time | 0.15 ms | 7.52 ms |

| Peak Inference Memory | 103.86 MB | 104.64 MB |

| Model Size | 1,055 bytes (1.03 KB) | 290,889 bytes (284.07 KB) |



\### Device Suitability

| Device Tier | Hardware Limits | Logistic Regression | Random Forest |

| :--- | :--- | :--- | :--- |

| \*\*Cloud\*\* | RAM >= 1 GB, Latency <= 100 ms, Size <= 500 MB | Yes | Yes |

| \*\*Edge\*\* | RAM 256-1024 MB, Latency <= 50 ms, Size <= 50 MB | Yes | Yes |

| \*\*Mobile\*\* | RAM 64-256 MB, Latency <= 20 ms, Size <= 10 MB | Yes | Yes |

| \*\*TinyML\*\* | RAM <= 256 KB, Latency <= 10 ms, Size <= 100 KB | Yes | \*\*No\*\* (File is too big) |



\## 4. Conclusions

1\. \*\*A bigger model does not always mean better accuracy:\*\* Logistic Regression got a higher score (0.9415) than Random Forest (0.9357). Because the dataset is small and clean, a simple linear model worked better without getting distracted by small noise in the data.

2\. \*\*Simple models make decisions much faster:\*\* Logistic Regression took only 0.15 ms to make a prediction on one patient, while Random Forest took 7.52 ms. Checking 100 decision trees is roughly 50 times slower than doing one basic math calculation.

3\. \*\*Model file size decides what hardware can run it:\*\* Logistic Regression is only 1.03 KB, so it fits easily onto small TinyML microchips. Random Forest is 284.07 KB, which fails the TinyML size budget (limit is 100 KB), meaning it can only run on phones, PCs, or cloud servers.

