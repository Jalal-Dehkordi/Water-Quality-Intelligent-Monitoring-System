# Water-Quality-Intelligent-Monitoring-System
Water Quality Intelligent Monitoring System

This project aims to develop an intelligent water quality monitoring system using machine learning techniques. The system analyzes a dataset from Kaggle to determine the safety of water for aquatic life.
Dataset

The dataset used for this project is available in the file WATERQUALITY.csv and contains various parameters related to water quality.
Features

The following features are used for analysis:

    pH
    Hardness
    Solids
    Chloramines
    Sulfate
    Conductivity
    Organic_carbon
    Trihalomethanes
    Turbidity
    Potability

Target Variable

    is_safe: Indicates whether the water is safe (1) or not (0) for aquatic life.

Data Pre-processing

    Handling missing values and converting data types.
    Oversampling the minority class using SMOTE to balance the dataset.
    Feature scaling using StandardScaler.

Models

Three models are trained and evaluated in this project:

    Artificial Neural Network (ANN)
    Convolutional Neural Network (CNN) - Model 1
    Convolutional Neural Network (CNN) - Model 2

Artificial Neural Network (ANN)

    Architecture:
        Input Layer
        Hidden Layer 1 (6 neurons, ReLU activation)
        Hidden Layer 2 (6 neurons, ReLU activation)
        Output Layer (1 neuron, Sigmoid activation)
    Optimizer: Adam
    Loss Function: Binary Cross-Entropy
    Metrics: Accuracy

Convolutional Neural Network (CNN) - Model 1

    Architecture:
        Conv1D Layer (64 filters, 2 kernel size, ReLU activation)
        MaxPooling1D
        Flatten
        Dense Layer (1 neuron, Sigmoid activation)
    Optimizer: Adam
    Loss Function: Binary Cross-Entropy
    Metrics: Accuracy

Convolutional Neural Network (CNN) - Model 2

    Architecture:
        Conv1D Layer 1 (32 filters, 2 kernel size, ReLU activation)
        MaxPooling1D
        Conv1D Layer 2 (64 filters, 2 kernel size, ReLU activation)
        MaxPooling1D
        Flatten
        Dense Layer (1 neuron, Sigmoid activation)
    Optimizer: Adam
    Loss Function: Binary Cross-Entropy
    Metrics: Accuracy

Evaluation Metrics

    Accuracy
    Confusion Matrix
    Classification Report (Precision, Recall, F1-score)

Instructions to Run the Code

    Clone the repository.
    Install the required libraries mentioned in the requirements.txt.
    Run the Jupyter notebook or Python script to execute the code.

Conclusion

The trained models can predict the safety of water for aquatic life based on the provided features. The performance of each model can be further improved by tuning hyperparameters or using additional features.
