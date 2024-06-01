# Diamond Price Prediction

## Project Overview

This project aims to predict the price of diamonds based on various attributes using machine learning techniques. The dataset contains information on nearly 54,000 diamonds, including features such as carat, cut, color, clarity, and price. The project involves data exploration, preprocessing, model training, and fine-tuning to achieve the best possible predictive performance. Finally, the results are tested and submitted to a Kaggle competition.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Explore the Data](#explore-the-data)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Fine-Tuning](#model-fine-tuning)
- [Testing and Submission](#testing-and-submission)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for this project is sourced from Kaggle and contains the following attributes for each diamond:

- **Carat**: Weight of the diamond
- **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **Color**: Diamond color, with D being the best and J the worst
- **Clarity**: A measurement of how clear the diamond is (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
- **Depth**: Total depth percentage
- **Table**: Width of the top of the diamond relative to the widest point
- **Price**: Price of the diamond
- **X**: Length in mm
- **Y**: Width in mm
- **Z**: Depth in mm

## Installation

To run this project, you'll need Python 3.x and the following libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/reham-shaban/Diamond-Price-Prediction.git
```

2. Open the Jupyter notebook:

```bash
jupyter notebook Diamond_price_prediction.ipynb
```

3. Run the cells in the notebook to explore the data, preprocess it, train models, and fine-tune the best model.

## Explore the Data

The initial phase of the project involves exploring the dataset to understand the distributions, relationships, and potential anomalies. Key steps include:

- **Loading the Data**: Import the dataset using pandas to inspect its structure and initial statistics.
- **Gather Information**: Exploring the columns information and the categorical values.
- **Visualizing the Distribution**: Plot the distribution of diamond prices using histograms to identify skewness and central tendency.
- **Box Plots**: Use box plots to detect outliers and understand the spread of data across different categories (cut, color, clarity).
- **Correlation Matrix**: Compute and visualize the correlation matrix to understand the linear relationships between numerical features.
- **Handling Missing Values**: Identify any missing values and decide on strategies to handle them (e.g., imputation or removal).
- **Handling Outliers**: Detect and handle outliers to ensure they do not adversely affect model performance.

The notebook provides detailed code and visualizations for these steps.

## Data Preparation

Data preparation involves transforming the raw dataset into a format suitable for machine learning algorithms. This includes:

- Encoding categorical features (cut, color, clarity)
- Scaling numerical features (carat, depth, table, dimensions)
- Splitting the dataset into training and testing sets

A pipeline is constructed using scikit-learn's `Pipeline` and `ColumnTransformer` to streamline these transformations.

## Model Training

Several machine learning models are trained on the prepared data:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

The performance of these models is evaluated using Mean Squared Error (MSE) metric.

## Model Fine-Tuning

The best-performing model is fine-tuned using Grid Search to optimize hyperparameters. This involves:

- Defining a grid of hyperparameters to search
- Using cross-validation to evaluate the model on different combinations of hyperparameters
- Selecting the combination that yields the best performance

## Testing and Submission

The predictions and results are submitted to the [Kaggle Diamond Price Prediction Competition 2024](https://www.kaggle.com/competitions/diamond-price-prediciton-2024/data) for evaluation.
