# SMOKER/DRINKER Classification
---

# Table of Contents
- [Introduction](#smokerdrinker-classification)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Testing the Prediction Service](#using-the-prediction-service)
# Introduction  
This project focuses on building a machine learning project to determine whether a person is a smoker/drinker from a set of numerical features about them.
---

# Getting Started
To run this project, follow these steps:

# Prerequisites

- Docker: You need to have Docker installed on your system.

# Installation

1. Clone this repository.

2. Build the Docker image using the following command:

```bash
docker build -t <image-name> .
```

3. Run the Docker container with the following command:

```bash
docker run -it --rm -p 4041:4041 <image-name>
    ```

4. Create and activate a virtual environment

```bash
pipenv shell
```
5. Install dependencies
```bash
pipenv install
```

# Using the Prediction Service
Once you have the container up and running, you can begin using the model by runing:
```bash
python test-predict.py
```
- Edit the dictionary in [test-predict.py](test-predict.py) with your custom records to get your predictions.
- You can also modify the `test-predict.py` file with samples from the [test_data](test_data.py) file.


- > [!IMPORTANT]
  > The prediction service can processe multiple records in a single request.