# CodeAlpha_CarPricePrediction

Task 3 of the CodeAlpha Data Science Internship: predict the selling price of a used car from its features (a **regression** problem).

## Dataset
`car_data.csv` has 301 used cars and 9 columns: `Car_Name`, `Year`, `Selling_Price` (target, in lakhs), `Present_Price` (price of the car when new, in lakhs), `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`.
There are no missing values. 2 duplicate rows were removed, which leaves 299 cars.

## Approach
1. **Cleaning**: removed duplicates and checked types and missing values.
2. **Feature engineering**
   - `Car_Age` = (newest year + 1) - `Year`, which is more meaningful than the raw year.
   - `Present_Price` and `Driven_kms` are heavily right-skewed (a few extreme cars, e.g. 92.6 lakhs and 500,000 km), so they were log-transformed.
   - `Car_Name` was dropped: 98 distinct models across about 300 rows is too sparse to encode reliably.
   - Categorical columns were one-hot encoded and numeric ones standardized, inside a pipeline.
3. **EDA**: price distribution, price vs present price and age, price by category, correlation matrix.
4. **Modeling**: 80/20 split. Five models were compared with **5-fold CV repeated 3 times** on the training set: Linear Regression, Linear Regression and Ridge on a **log-transformed target**, Random Forest and Gradient Boosting.
5. **Evaluation** on the held-out test set with R², MAE and RMSE, plus permutation feature importance.

## Results
| Model | CV R2 | Test R2 | Test MAE (lakhs) |
|---|---|---|---|
| Linear Regression | 0.810 | 0.487 | 2.16 |
| **Linear Regression (log target)** | **0.957** | **0.890** | **0.97** |
| Ridge (log target) | 0.955 | 0.891 | 0.96 |
| Random Forest | 0.891 | 0.531 | 1.47 |
| Gradient Boosting | 0.881 | 0.724 | 1.19 |

The selected model (Linear Regression on log price) predicts selling price with an average error of about **0.97 lakhs** on unseen cars, and explains about 89% of the variance on the test set.

**What drives the price**
- The **new price of the car** (`Present_Price`) is by far the strongest factor, followed by **car age**.
- Fuel type, kilometres driven, seller type, transmission and number of owners matter much less in this dataset.

## Notes and limitations
- Modeling the log of the price was the biggest improvement. It fits how depreciation works (a percentage drop rather than a fixed amount) and handles the skewed prices.
- Tree models scored worse on the test set. They cannot predict outside the price range seen in training, and the test set contains an expensive car (33 lakhs). Linear models extrapolate better here.
- Test R2 (0.89) is lower than CV R2 (0.96). The test set is only 60 cars, so a few errors matter a lot. The largest errors are old, high-mileage cars with an expensive original price, where the model overpredicts. Brand and model (dropped, see above) likely explain part of that.
- With about 300 cars, treat the numbers as indicative, not exact. Only 2 cars are CNG, so the model cannot learn anything reliable about CNG.

## Run
```bash
pip install -r requirements.txt
python car_price_prediction.py
```

## Files
- `car_price_prediction.py`: full pipeline
- `model_comparison.csv`: model scores
- `images/`: plots
