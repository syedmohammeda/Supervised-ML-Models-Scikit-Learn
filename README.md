# Week 2 – Supervised Machine Learning Models

## 1. Project Overview
This project demonstrates supervised machine learning for both regression and classification tasks using Python and Scikit-learn. The workflow covers data loading, preprocessing, model training, prediction, evaluation, visualization, and documentation.

## 2. Objectives
- Build and evaluate supervised learning models.
- Compare regression and classification approaches.
- Train and assess multiple algorithms.
- Interpret results using appropriate metrics.
- Produce reusable, well-structured Python code.

## 3. Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 4. Dataset Description
The project uses the wine quality dataset, which contains physicochemical properties of wine and a quality score used as the regression target. For classification, the quality score is converted into a binary label using a threshold of 6: wines with quality >= 6 are labeled as 1 (higher quality), and wines below 6 are labeled as 0 (lower quality).

## 5. Project Structure
```text
week2_supervised_ml/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── week2_supervised_ml.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── regression_models.py
│   ├── classification_models.py
│   └── evaluation.py
├── results/
│   ├── model_comparison.csv
│   ├── regression_results.csv
│   ├── classification_results.csv
│   └── figures/
├── requirements.txt
├── README.md
├── main.py
└── .venv/
```

## 6. Data Preprocessing
- Load the dataset using Pandas.
- Inspect the first rows, shape, column names, and dtypes.
- Check for missing values and duplicates.
- Fill missing numeric values with medians.
- Remove duplicate rows.
- Split data into training and testing sets with `random_state=42`.
- Apply standard scaling where required, especially for Logistic Regression and KNN.

## 7. Regression Models
The regression task predicts the numeric wine quality score using:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Regression performance is evaluated using:
- MAE
- MSE
- RMSE
- R² Score

## 8. Classification Models
The classification task predicts whether a wine is higher quality or lower quality using:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- KNN Classifier

Classification performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix

## 9. Evaluation Metrics
Different metrics measure different aspects of prediction quality:
- MAE measures average absolute error.
- RMSE emphasizes larger errors.
- R² indicates how much variance is explained by the model.
- Accuracy measures overall correctness.
- Precision measures false positives.
- Recall measures false negatives.
- F1 balances precision and recall.

## 10. Results
Results are generated and saved automatically when you run:

```bash
python main.py
```

## 11. Visualizations
The project saves plots to `results/figures/` including:
- Target distribution
- Correlation heatmap
- Regression actual-vs-predicted plot
- Regression model comparison plot
- Classification confusion matrices
- Classification accuracy comparison
- Classification F1 comparison

## 12. How to Run the Project
### Create a virtual environment
```bash
python -m venv .venv
```

### On Windows
```bash
.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the project
```bash
python main.py
```

### Launch the notebook
```bash
jupyter notebook
```

## 13. Installation Instructions
1. Open the project folder in VS Code.
2. Create and activate the virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Run the script or open the notebook.

## 14. Key Learnings
- Supervised learning learns from labeled data.
- Regression predicts continuous values while classification predicts class labels.
- Train/test splits help evaluate generalization.
- Scaling improves models such as Logistic Regression and KNN.
- Model comparison should use appropriate metrics for each task.

## 15. Conclusion
This project demonstrates the core concepts of supervised machine learning, including model training, evaluation, and comparison. The workflow is designed to be reproducible, easy to follow, and suitable for a week 2 internship assignment.

## 16. Final Results Table
### Regression
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.504141 | 0.656513 | 0.391536 |
| Decision Tree Regressor | 0.650735 | 0.945189 | -0.261206 |
| Random Forest Regressor | 0.467445 | 0.617338 | 0.461984 |

### Classification
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.735294 | 0.760870 | 0.729167 | 0.744681 |
| Decision Tree Classifier | 0.625000 | 0.641892 | 0.659722 | 0.650685 |
| Random Forest Classifier | 0.757353 | 0.786765 | 0.743056 | 0.764286 |
| KNN Classifier | 0.687500 | 0.690323 | 0.743056 | 0.715719 |

The random forest regressor achieved the lowest RMSE and the highest R² among the regression models, while the random forest classifier achieved the best accuracy and F1 score among the classification models. These differences are expected because each metric measures a different aspect of model performance: RMSE focuses on prediction error magnitude, R² measures explained variance, and classification metrics evaluate correctness and class-balance trade-offs.
