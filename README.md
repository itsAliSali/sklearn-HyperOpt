# sklearn-HyperOpt

## Overview

**sklearn-HyperOpt** is a Jupyter Notebook-based project that automates hyperparameter optimization and feature selection for various machine learning algorithms using Python's scikit-learn library. This project is designed to help data scientists and ML practitioners quickly find optimal hyperparameters and select relevant features, enhancing model performance with minimal manual effort.

To demonstrate the project’s capabilities, the [Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UC Irvine’s Machine Learning Repository is used. The dataset includes two datasets related to red and white variants of Portuguese "Vinho Verde" wine, featuring physicochemical properties and sensory ratings as output variables.

## Features

- **Hyperparameter Optimization**: Efficiently tunes the hyperparameters of multiple machine learning models.
- **Feature Selection Methods**: Implements several feature selection techniques to identify the most informative features.

## Project Structure

- `wine_quality.ipynb`: Main Jupyter Notebook containing the code for hyperparameter optimization, explanations, and results.
- `feature_selection.ipynb`:  Jupyter Notebook dedicated to various feature selection methods.

## Supported Algorithms and Hyperparameter Grids

The project includes hyperparameter optimization for several machine learning algorithms using the following configurations:

1. **Linear Regression**: Option for intercept fitting.
2. **Decision Tree Regressor**: Criteria options, minimum samples per split, maximum features, and minimum samples per leaf.
3. **Random Forest Regressor**: Tunable parameters for estimators, features, depth, and sample splits and leaf sizes.
4. **Gradient Boosting**: Gradient boosting parameters such as estimators, features, subsample, sample splits, and learning rates.
5. **MLP Regressor**: Configurations for layer sizes, learning rate, regularization (alpha), momentum, activation functions, and solvers.
6. **Gaussian Process Regressor**: Adjustable alpha (diagonal bias) and optimizer restarts.
7. **k-Nearest Neighbors (kNN) Regressor**: Configurable neighbor count, weight method, algorithm choice, and leaf size.

These configurations aim to provide a comprehensive search space for each model, leveraging scikit-learn's `RandomizedSearchCV` for efficient hyperparameter tuning.

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