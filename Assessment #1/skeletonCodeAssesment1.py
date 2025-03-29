import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV


# Load appointment data
df = pd.read_csv("appointments.csv")  # Contains scheduled_time, actual_time, doctor_id, patient_id

# Feature Engineering
df['delay'] = (pd.to_datetime(df['actual_time']) - pd.to_datetime(df['scheduled_time'])).dt.total_seconds() / 60
df['hour'] = pd.to_datetime(df['scheduled_time']).dt.hour
df['day_of_week'] = pd.to_datetime(df['scheduled_time']).dt.dayofweek

# Define features and target variable
features = ['doctor_id', 'hour', 'day_of_week']
target = 'delay'

# splitting data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning to get best parameter of model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
model = RandomForestRegressor()
model.fit(X, y)


# Predict delay for future appointments
def predict_wait_time(doctor_id, scheduled_time):
    hour = scheduled_time.hour
    day_of_week = scheduled_time.weekday()
    return model.predict([[doctor_id, hour, day_of_week]])[0]  # Predicted delay in minutes

def assign_patient_to_doctor(patient_info, available_doc):
    predictions = {}
    for doctor in available_doc:
        patient_info['doctor_id'] = doctor
        predicted_wait = model.predict(doctor,patient_info['scheduled_time'])
        predictions[doctor] = predicted_wait
    
    best_doctor = min(predictions, key=predictions.get)
    return best_doctor, predictions[best_doctor]


def notify_patient(predicted_wait_time,doctor=None):
    if doctor:                                            # if doctor reassigned in can of early arrivals
        return f"Reassigned to Doctor {doctor} with updated wait time: {predicted_wait_time:.2f} minutes."
    if predicted_wait_time < 30:
        return f"Your estimated wait time is {predicted_wait_time} minutes."
    else:
        return "Your estimated wait time is over 30 minutes. Please plan accordingly. "

# Example usage
scheduled_time = datetime(2024, 3, 26, 18, 30)
arrival_time = datetime(2024, 3, 26, 17, 30)      #we get info about arrival time maybe with scanning qr code before entering clinic or manually by stuff into app
available_doctors = [1, 2, 3, 8, 10]

# Predict initial wait time
predicted_delay = predict_wait_time(doctor_id=5, scheduled_time=scheduled_time)
print(notify_patient(predicted_delay))

# Handle early arrivals (> 10 minutes early)
early_arrival_threshold = 10  # Minutes
early_arrival_time = (scheduled_time - arrival_time).total_seconds() / 60

if early_arrival_time > early_arrival_threshold:
    best_doctor, updated_wait_time = assign_patient_to_doctor(scheduled_time=arrival_time, available_doctors=available_doctors)
    notify_patient(updated_wait_time,best_doctor)
