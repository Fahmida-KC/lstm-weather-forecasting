# LSTM-Based Weather Time-Series Forecasting


## Project Overview

This project develops a **Long Short-Term Memory (LSTM)** neural network to forecast air temperature based on historical weather observations.

The model uses a sliding-window approach to learn temporal patterns from consecutive temperature measurements and predict the next temperature value. The implementation uses Python and TensorFlow/Keras.

## Dataset Description

The project uses the `cleaned_weather.csv` dataset, which contains historical meteorological observations.

| Attribute                | Description                        |
| ------------------------ | ---------------------------------- |
| Total observations       | 52,696                             |
| Meteorological variables | 20                                 |
| Recording frequency      | 10 minutes                         |
| Target variable          | Air temperature (`T`)              |
| Forecasting type         | Univariate time-series forecasting |

**Note:** Although the original dataset contains 20 meteorological variables, only air temperature was selected for this implementation.

## Methodology

The project follows these major steps:

1. **Data Loading and Exploration:** Imported the dataset using Pandas and examined its structure and temperature trends.
2. **Sequence Generation:** Applied a sliding-window technique to convert the time series into supervised learning sequences.
3. **Data Splitting:** Divided the dataset chronologically into training, validation, and testing sets.
4. **Model Development:** Designed an LSTM-based neural network using TensorFlow/Keras.
5. **Model Training:** Trained the network using the Adam optimizer and Mean Squared Error loss.
6. **Prediction and Evaluation:** Generated predictions and compared them with actual temperature observations.

### Sliding-Window Approach

A window size of 5 was used to generate input sequences.

```python
Input:  [T(t-4), T(t-3), T(t-2), T(t-1), T(t)]
Output: T(t+1)
```

The model predicts the next 10-minute temperature observation based on the previous five observations.

### Dataset Splitting

| Dataset    |    Samples |
| ---------- | ---------: |
| Training   |     40,000 |
| Validation |      5,000 |
| Testing    |      7,691 |
| **Total**  | **52,691** |

The data was split chronologically without random shuffling to preserve temporal order.

## LSTM Model Architecture

The neural network consists of the following layers:

| Layer  | Configuration               |
| ------ | --------------------------- |
| Input  | 5 time steps, 1 feature     |
| LSTM   | 64 units                    |
| Dense  | 8 neurons, ReLU activation  |
| Output | 1 neuron, Linear activation |

**Total trainable parameters: 17,425**

### Training Configuration

| Hyperparameter    | Value                          |
| ----------------- | ------------------------------ |
| Optimizer         | Adam                           |
| Learning rate     | 0.0001                         |
| Loss function     | Mean Squared Error (MSE)       |
| Evaluation metric | Root Mean Squared Error (RMSE) |
| Epochs            | 10                             |
| Model checkpoint  | Best validation-loss model     |

## Results

The model showed a reduction in training and validation errors during the training process.

| Epoch | Training RMSE | Validation RMSE |
| ----- | ------------: | --------------: |
| 1     |       10.5425 |          0.3577 |
| 5     |        0.3794 |          0.2042 |
| 8     |        0.2539 |          0.1779 |
| 10    |        0.2436 |          0.1741 |

### Best Validation Performance

**Validation RMSE: 0.1741**

The model predictions were visualized against actual temperature observations for the training, validation, and testing datasets.

*Note: Final test-set evaluation metrics have not yet been calculated.*

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Development Environment:** Kaggle Notebook

## Repository Structure

```text
lstm-weather-forecasting/
│
├── weather_forecasting.ipynb
├── README.md
└── requirements.txt
```

## Getting Started

Download `cleaned_weather.csv` from the original Kaggle dataset and update the file path in the notebook if necessary.

Open `weather_forecasting.ipynb` using Jupyter Notebook, Google Colab, or Kaggle and execute the cells.

## Future Work

- Extend the model to multivariate weather forecasting using additional meteorological variables.
- Investigate different input-window sizes and forecasting horizons.
- Compare LSTM with GRU and other deep learning architectures.
- Apply systematic hyperparameter optimization.
- Evaluate test performance using MAE, RMSE, and R².
- Explore multi-step temperature forecasting.

##
