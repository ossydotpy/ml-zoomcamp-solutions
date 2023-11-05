# Breast Cancer Survival Prediction
---
## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution](#problem-statement)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Testing the Prediction Service](#using-the-prediction-service)

## Problem Statement
Breast cancer is one of the most prevalent and life-threatening diseases among women worldwide. Early detection is vital, as it can lead to more effective treatment options and improved patient outcomes. Accurate prediction of breast cancer survival is crucial. In this project, we aim to develop a predictive model that can determine whether a breast cancer patient will survive or not based on a set of relevant features.

## Solution
Our solution involves building a machine learning model using Python to predict breast cancer survival. We will utilize a dataset containing various patient features, such as age, tumor size, lymph node information, and more, to train and evaluate our predictive model. We will consider metrics like accuracy, recall, and precision to assess the model's performance.

You can find the dataset used [here](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer/data)

## Getting Started
TTo run this project, follow these steps:

### Prerequisites

- Docker: You need to have Docker installed on your system.

### Installation

1. Clone this repository.

2. Build the Docker image using the following command:

    ```bash
    docker build -t <image-name> .
    ```

3. Run the Docker container with the following command:

    ```bash
    docker run -it --rm -p 4041:4041 <image-name>
    ```

4. Install development packages with the following command:

    ```bash
    pipenv install --dev
    ```

### Using the Prediction Service
Once you have the container up and running, you can begin using the model by runing:
```bash
python test-predict.py
```
- Edit the dictionary in [test-predict.py](test-predict.py) with your custom records to get your predictions.
- You can also modify the `test-predict.py` file with samples from the [patient-test-dataset](Datasets/patient-test-dataset) file.
- > [!NOTE]
  > The prediction service processes one record in a single request.

> [!WARNING]
> This model is for educational purposes only and may not provide accurate or reliable predictions. Please use caution and do not make critical decisions based solely on its outputs. Always consult with domain experts and consider real-world data before relying on this model for practical applications.