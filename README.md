# Titanic Survival Prediction

This project uses the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project in machine learning with readily available data.

## Project Structure

- `titanic_survival_prediction.py`: Main script to load data, preprocess, train, and evaluate the model.
- `README.md`: Project documentation.
- `catboost_titanic_model.cbm`: Saved CatBoost model.

## Dataset

The dataset typically contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived. The dataset is available at: [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/titanic-survival-prediction.git
    cd titanic-survival-prediction
    ```

2. Install required libraries:
    ```sh
    pip install pandas catboost scikit-learn matplotlib
    ```

## Usage

1. Run the script to train the model and visualize the results:
    ```sh
    python titanic_survival_prediction.py
    ```

2. The script will load the dataset, preprocess the data, train a CatBoost classifier, and display the following:
   - Cross-validation plot
   - Training process plot
   - Feature importance plot
   - Confusion matrix
   - ROC curve

## Data Preprocessing

- Fill missing values for `Age`, `Embarked`, and `Fare`.
- Convert categorical features (`Sex`, `Embarked`) to numeric values.
- Drop unnecessary columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).

## Model

The model used in this project is a [CatBoostClassifier](https://catboost.ai/). CatBoost is a high-performance open-source library for gradient boosting on decision trees.

### Model Parameters

- `iterations`: 1000
- `depth`: 6
- `learning_rate`: 0.1
- `loss_function`: 'Logloss'
- `early_stopping_rounds`: 50

## Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Confusion Matrix
- ROC Curve

## Results

The model achieved a high accuracy on the test set. The feature importance plot indicates which features are most influential in predicting survival. The confusion matrix and ROC curve provide additional insights into the model's performance.

## Visualization

- **Confusion Matrix**: Displays the number of true positive, true negative, false positive, and false negative predictions.
- **ROC Curve**: Plots the true positive rate against the false positive rate at various threshold settings.

## Save and Load Model

The trained model is saved as `catboost_titanic_model.cbm`. You can load the model using:

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('catboost_titanic_model.cbm')
