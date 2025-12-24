# 🛡️ Deep Learning-Driven Heuristic Intrusion Detection System (DL-HIDS)

This project implements a **Heuristic Intrusion Detection System (IDS)** designed specifically for **Software-Defined Networks (SDN)**. By leveraging Deep Learning (DL) and Machine Learning (ML) techniques, the system is capable of detecting network anomalies and classifying various types of cyber attacks with high accuracy.

## 📂 Project Overview

Traditional IDS often fail to detect zero-day attacks and suffer from high false-negative rates. This project proposes a **DL-HIDS** that combines heuristic analysis with deep neural networks to address these limitations.

**Key Objectives:**
* **Traffic Analysis**: Preprocessing and analyzing network traffic data.
* **Anomaly Detection**: Identifying malicious patterns in SDN environments.
* **Model Comparison**: Benchmarking various ML classifiers against Deep Learning models.
* **Performance Optimization**: Achieving high precision and recall for robust security.

## 📊 Dataset

The project utilizes the **SDN Intrusion Dataset** (likely based on the **InSDN** benchmark dataset), which contains distinct traffic patterns for normal and malicious behavior.

**Key Features Analyzed:**
The dataset typically includes flow-based features such as:
* Protocol types (TCP, UDP, ICMP)
* Packet counts and byte counts
* Duration of flows
* Source and Destination IP/Ports

## 🛠️ Technologies & Libraries Used

The project is implemented in Python. Key libraries and frameworks include:

* **Deep Learning**:
    * `TensorFlow` / `Keras`: For building the Sequential Neural Network (Dense layers).
* **Machine Learning**:
    * `Scikit-learn`: For implementing Random Forest, Decision Tree, and Logistic Regression.
    * `LazyPredict`: For rapid prototyping and benchmarking of multiple classifiers.
* **Data Manipulation**:
    * `Pandas` & `NumPy`: For data loading, cleaning, and transformation.
* **Visualization**:
    * `Matplotlib` & `Seaborn`: For plotting training accuracy/loss curves and confusion matrices.

## 🧠 Models Implemented

The notebook explores a variety of algorithms to find the best fit for intrusion detection:

1.  **Deep Learning Model**:
    * A **Sequential Deep Neural Network (DNN)** with multiple dense layers.
    * Uses `ReLU`, `Tanh`, `Sigmoid`, and `Softmax` activation functions.
    * Optimized using the `Adam` optimizer.
2.  **Machine Learning Classifiers**:
    * **Random Forest Classifier**
    * **Decision Tree Classifier**
    * **Logistic Regression**
    * **Bagging Classifier**

## 📈 Results & Visualizations

The analysis includes:
* **Training vs. Validation Accuracy**: Plots showing the learning curve of the deep learning model over epochs.
* **Model Benchmarking**: Using `LazyClassifier` to compare performance metrics (Accuracy, F1-Score) across dozens of models.
* **Confusion Matrices**: To visualize false positives and true positives.

## 🚀 How to Run

1.  Clone this repository.
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy tensorflow scikit-learn matplotlib seaborn lazypredict
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Heuristic Intrusion Detection System.ipynb"
    ```
4.  **Dataset Setup**: Ensure the dataset file (e.g., `SDN_Intrusion.csv`) is available.
    * *Note*: The code currently looks for the dataset at `/content/SDN_Intrusion.csv`. We may need to update the `pd.read_csv()` path in the notebook to match our local file location.
  
## Results

**Detection Rates:**

- Baseline Model Accuracy: 70.4 %

- Final Smart Model Accuracy: 96.7 %
