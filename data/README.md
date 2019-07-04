# data

Pima Indians Diabetes Database. 768 rows, 9 columns.

Columns:
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome (0 or 1)

Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database

The CSV file `diabetes.csv` is the raw download from Kaggle.

Note: Glucose, BloodPressure, SkinThickness, Insulin, and BMI have zero values
that are really missing. The preprocessing step replaces them with NaN and
imputes the column median.
