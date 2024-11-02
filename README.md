# sklearn-HyperOpt

## Overview

**sklearn-HyperOpt** is a Jupyter Notebook-based project that automates hyperparameter optimization and feature selection for various machine learning algorithms using Python's `scikit-learn` library. This project is designed to help data scientists and ML practitioners quickly find optimal hyperparameters and select relevant features, enhancing model performance with minimal manual effort.

To demonstrate the project’s capabilities, the [Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UC Irvine’s Machine Learning Repository is used. The dataset includes two datasets related to red and white variants of Portuguese "Vinho Verde" wine, featuring physicochemical properties and sensory ratings as output variables. Due to privacy restrictions, details such as grape types or wine brand are unavailable. For more on the dataset, see [Cortez et al., 2009](http://www.vinhoverde.pt/en/).

## Features

- **Hyperparameter Optimization**: Efficiently tunes the hyperparameters of multiple machine learning models.
- **Feature Selection Methods**: Implements several feature selection techniques to identify the most informative features.
- **Compatibility with scikit-learn**: Uses built-in methods from the `scikit-learn` library to perform both optimization and feature selection.
- **Jupyter Notebook**: Code and outputs are fully documented in a Jupyter Notebook, making it easy to follow and adapt.

## Project Structure

- `HyperOpt_ML.ipynb`: Main Jupyter Notebook containing the code, explanations, and results.

## Supported Algorithms and Hyperparameter Grids

The project includes hyperparameter optimization for several machine learning algorithms using the following configurations:

1. **Linear Regression**: Options for intercept fitting.
2. **Decision Tree Regressor**: Criteria options, minimum samples per split, maximum features, and minimum samples per leaf.
3. **Random Forest Regressor**: Tunable parameters for estimators, features, depth, and sample splits and leaf sizes.
4. **Gradient Boosting (MultiOutput Regressor)**: Gradient boosting parameters such as estimators, features, subsample, sample splits, and learning rates.
5. **MLP Regressor**: Configurations for layer sizes, learning rate, regularization (`alpha`), momentum, activation functions, and solvers.
6. **Gaussian Process Regressor**: Adjustable alpha and optimizer restarts.
7. **k-Nearest Neighbors (kNN) Regressor**: Configurable neighbor count, weight method, algorithm choice, and leaf size.

These configurations aim to provide a comprehensive search space for each model, leveraging `scikit-learn`'s `RandomizedSearchCV` for efficient hyperparameter tuning.

## Prerequisites

- **Python 3.x**: This project requires Python 3.x.
- **Libraries**:
  - `pandas`, `numpy` for data handling
  - `scikit-learn` for machine learning models, optimization, and feature selection
  - `matplotlib`, `seaborn` for data visualization
  - `scipy` for statistical functions

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy