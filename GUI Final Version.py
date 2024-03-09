import pickle
import tkinter as tk
from tkinter import messagebox
import unicodeit
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score,
                             median_absolute_error, mean_squared_log_error, root_mean_squared_error)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from ngboost import NGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

def safe_msle(y_true, y_pred):
    if np.any(y_true < 0) or np.any(y_pred < 0):
        return np.nan
    return mean_squared_log_error(y_true, y_pred)

# Load your dataset
data = pd.read_excel("Extended_Dataset.xlsx", sheet_name='data')

# Assuming the last column is the target and the others are features
X = data.iloc[:, :-1]
# X = X.drop(columns=['Aso (mm2)'])
y = data.iloc[:, -1]

# Define base models
base_models = [
    ('lr', LinearRegression()),
    ('svr', SVR()),
    ('kr', KernelRidge()),
    ('gpr', GaussianProcessRegressor()),
    ('dtr', DecisionTreeRegressor()),
    ('rfr', RandomForestRegressor()),
    ('gbr', GradientBoostingRegressor()),
    ('xgbr', xgb.XGBRegressor()),
    ('ngbr', NGBRegressor()),
    ('mlpr', MLPRegressor()),
]

# Filter out ('ngbr', NGBRegressor()) from base_models when defining stacked_model
filtered_base_models = [model for model in base_models if model[0] != 'ngbr']
# Define Stacking Regressor
stacked_model = StackingRegressor(
    estimators=filtered_base_models,
    final_estimator=LinearRegression(), cv=None, passthrough=True
)

# Include the stacking regressor in the models list
models = base_models + [('stacked', stacked_model)]
scaler = MinMaxScaler()
def cv(model):
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores_train = []
    scores_test = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Feature scaling

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        scores_train.append([
            r2_score(y_train, y_pred_train), #r2
            root_mean_squared_error(y_train, y_pred_train), #rmse
            safe_msle(y_train, y_pred_train), # msle
            mean_absolute_error(y_train, y_pred_train), # mae
            mean_absolute_percentage_error(y_train, y_pred_train), # mape
            median_absolute_error(y_train, y_pred_train), # mae_1
            root_mean_squared_error(y_train, y_pred_train) / np.mean(y_train)
        ])

        scores_test.append([
            r2_score(y_test, y_pred_test),
            root_mean_squared_error(y_test, y_pred_test),
            safe_msle(y_test, y_pred_test),
            mean_absolute_error(y_test, y_pred_test),
            mean_absolute_percentage_error(y_test, y_pred_test),
            median_absolute_error(y_test, y_pred_test),
            root_mean_squared_error(y_test, y_pred_test) / np.mean(y_test)
        ])

    return np.mean(scores_train, axis=0), np.mean(scores_test, axis=0)

# Update the loop to iterate over the models including the stacked model
model_train_scores = []
model_test_scores = []
for name, model in models:

    scores_train, scores_test = cv(model)
    model_test_scores.append(scores_test)
    model_train_scores.append(scores_train)

model_train_scores = np.array(model_train_scores)
model_test_scores = np.array(model_test_scores)

# Proceed with the remaining part of your code to calculate CPI scores and export results


def calculate_cpi(scores):
    # Assuming that higher scores are better. If not, adjust the scores accordingly before this function.

    # Handle NaN values (simple imputation with the mean for each metric)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scores_imputed = imputer.fit_transform(scores)

    # Normalize the scores for each metric across all models
    # The normalization is such that it reflects higher values indicating better performance.
    normalized_scores = (scores_imputed - np.min(scores_imputed, axis=0)) / (
                np.max(scores_imputed, axis=0) - np.min(scores_imputed, axis=0))

    # Compute the CPI for each model as the mean of the normalized scores
    cpi_scores = np.mean(normalized_scores, axis=1)

    return cpi_scores




# Calculate MDS-CPI scores
cpi_scores_train = calculate_cpi(model_train_scores)
cpi_scores = calculate_cpi(model_test_scores)




# Save the trained model
with open('stacked_model.pkl', 'wb') as model_file:
    pickle.dump(stacked_model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
# tkinter GUI
root = tk.Tk()
root.title(f"Prediction of load carrying capacity of column")

canvas1 = tk.Canvas(root, width=550, height=600)
canvas1.configure(background='#e9ecef')
canvas1.pack()

label0 = tk.Label(root, text='Load carrying capacity of column', font=('Times New Roman', 15, 'bold'), bg='#e9ecef')
canvas1.create_window(70, 20, anchor="w", window=label0)

label_phd = tk.Label(root, text='Developed by: Mr. Rupesh Kumar Tipu\n K. R. Mangalam University, India.\n '
                                'tipu0003@gmail.com',
                     font=('Futura Md Bt', 12), bg='#e9ecef')

canvas1.create_window(100, 60, anchor="w", window=label_phd)

label_input = tk.Label(root, text='Input Variables', font=('Times New Roman', 12, 'bold', 'italic', 'underline'),
                       bg='#e9ecef')
canvas1.create_window(20, 90, anchor="w", window=label_input)

# Labels and entry boxes
labels = ['Area of concrete portions  (mm\u00b2)',
          'Conrete strength of standard cylinders  (MPa)',
          'Area of internal steel tube  (mm\u00b2)',
          'Yield strength of internal steel tube  (MPa)',
          'Area of external steel tube (mm\u00b2)',
          'Yield strength of external steel tube   (MPa)',
          'Eccentric loading ratio  (e/(2b))',

          ]

entry_boxes = []
for i, label_text in enumerate(labels):
    label = tk.Label(root, text=unicodeit.replace(label_text), font=('Times New Roman', 12, 'italic'), bg='#e9ecef',
                     pady=5)
    canvas1.create_window(20, 120 + i * 30, anchor="w", window=label)

    entry = tk.Entry(root)
    canvas1.create_window(480, 120 + i * 30, window=entry)
    entry_boxes.append(entry)

# label_output = tk.Label(root, text='Flow of Concrete', font=('Times New Roman', 12, 'bold'),
# bg='#e9ecef')
# canvas1.create_window(50, 420, anchor="w", window=label_output)

label_output1 = tk.Label(root, text='Load carrying capacity:', font=('Times New Roman', 18, 'bold'),
                         bg='#e9ecef')
canvas1.create_window(20, 560, anchor="w", window=label_output1)

def reset_entries():
    for entry in entry_boxes:
        entry.delete(0, tk.END)
def values():
    # Validate and get the values from the entry boxes
    input_values = []
    for entry_box in entry_boxes:
        value = entry_box.get().strip()
        if value:
            try:
                input_values.append(float(value))
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")
                return
        else:
            messagebox.showerror("Error", "Please fill in all the input fields.")
            return

    # Calculate additional features
    Ac_x_Aso = (input_values[0]*input_values[4])
    fc_x_Aso = (input_values[1]*input_values[4])
    Asi_x_fyi = input_values[2] * input_values[3]

    # Append additional features to the input_values list
    input_values.extend([Ac_x_Aso, fc_x_Aso, Asi_x_fyi])

    # If all input values are valid, proceed with prediction
    print(len(input_values))
    input_data = pd.DataFrame([input_values ],
                        columns=X.columns)

    # Load the trained MultiOutputRegressor model
    # Assuming input_values are collected correctly
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('stacked_model.pkl', 'rb') as model_file:
        trained_model = pickle.load(model_file)

    input_data = pd.DataFrame([input_values], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)  # Scale the data
    prediction_result = trained_model.predict(input_data_scaled)
    prediction_result1 = round(prediction_result[0], 2)

    # Display the prediction on the GUI
    label_prediction = tk.Label(root, text=f'{str(prediction_result1)} kN', font=('Times New Roman', 20, 'bold'),
                                bg='white')
    canvas1.create_window(280, 560, anchor="w", window=label_prediction)

    # label_prediction1 = tk.Label(root, text=f'{str(prediction_result2)} MPa', font=('Times New Roman', 20, 'bold'),
    #                              bg='white')
    # canvas1.create_window(230, 500, anchor="w", window=label_prediction1)


button1 = tk.Button(root, text='Predict', command=values, bg='#4285f4', fg='white',
                    font=('Times New Roman', 20, 'bold'),
                    bd=3, relief='ridge')
canvas1.create_window(440, 560, anchor="w", window=button1)

# Reset Button
button_reset = tk.Button(root, text="Reset", command=reset_entries, bg="red", fg="white", font=("Times New Roman", 20, "bold"), bd=3, relief="ridge")
canvas1.create_window(500, 500, window=button_reset)

root.mainloop()
