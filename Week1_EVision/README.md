# EVision
EVision explores electric vehicle data to identify factors influencing EV prices and sales trends. It focuses on data cleaning, preprocessing, and exploratory data analysis (EDA) using Python, Pandas, NumPy, Matplotlib, and Seaborn, forming the foundation for future predictive modeling
# EV Price Prediction – Task 1

## Objective
Develop a machine learning model to predict the price of an Electric Vehicle (EV) based on its specifications and performance metrics.

## Dataset
- Source: Custom dataset (`car_price_prediction_.csv`) provided by the user.
- Preprocessed to focus on **Electric** and **Hybrid** vehicles (treated as EVs for this task).
- Final cleaned dataset: **~850 EV/Hybrid entries**.
- Features used: `Brand`, `Year`, `Engine Size`, `Fuel Type`, `Transmission`, `Mileage`, `Condition`, `Model`.
- Target: `Price`

## Key EDA Insights
- Strong correlation between `Year` and `Price` (newer cars = higher price).
- `Mileage` shows a moderate negative correlation with `Price`.
- Tesla, BMW, and Toyota dominate the EV/Hybrid segment in the dataset.
- Price distribution is right-skewed; log-transformation improves model performance.

## Methodology
1. **Data Cleaning**: Remove irrelevant fuel types (Petrol/Diesel-only), handle categorical variables.
2. **Feature Engineering**: One-hot encoding for `Brand`, `Model`, `Condition`, etc.
3. **Model**: Random Forest Regressor (handles non-linearity and interactions well).
4. **Evaluation**: RMSE and R² score on test set.

## Results (Baseline)
- **R² Score**: ~0.85  
- **RMSE**: ~\$8,500  

> Note: Performance can be improved with better features (e.g., battery capacity, range), but these are not present in the current dataset.

## How to Run
1. Clone this repository.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt