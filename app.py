from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Generate synthetic data for training
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'current_sugar': np.random.rand(n_samples) * 200 + 100,
        'Csystolic_bp': np.random.rand(n_samples) * 60 + 120,
        'Cdiastolic_bp': np.random.rand(n_samples) * 40 + 80,
        'previous_sugar': np.random.rand(n_samples) * 200 + 100,
        'Psystolic_bp': np.random.rand(n_samples) * 60 + 120,
        'Pdiastolic_bp': np.random.rand(n_samples) * 40 + 80
    }
    data['systolic_bp'] = 0.4 * data['Csystolic_bp'] + 0.3 * data['Psystolic_bp'] + np.random.randn(n_samples) * 8
    data['diastolic_bp'] = 0.4 * data['Cdiastolic_bp'] + 0.2 * data['Pdiastolic_bp'] + np.random.randn(n_samples) * 5
    data['sugar_level'] = 0.5 * data['current_sugar'] + 0.3 * data['previous_sugar'] + np.random.randn(n_samples) * 7
    return pd.DataFrame(data)

def prepare_data(df):
    features = ['current_sugar', 'Csystolic_bp', 'Cdiastolic_bp', 'previous_sugar', 'Psystolic_bp', 'Pdiastolic_bp']
    target_systolic_bp = 'systolic_bp'
    target_diastolic_bp = 'diastolic_bp'
    target_sugar_level = 'sugar_level'
    X = df[features]
    y_systolic_bp = df[target_systolic_bp]
    y_diastolic_bp = df[target_diastolic_bp]
    y_sugar_level = df[target_sugar_level]
    return X, y_systolic_bp, y_diastolic_bp, y_sugar_level

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.2f}")
    print(f"Model R^2 Score: {r2:.2f}")
    return model

# Generate synthetic data and train models globally
df = generate_synthetic_data()
X, y_systolic_bp, y_diastolic_bp, y_sugar_level = prepare_data(df)
model_systolic_bp = train_random_forest(X, y_systolic_bp)
model_diastolic_bp = train_random_forest(X, y_diastolic_bp)
model_sugar_level = train_random_forest(X, y_sugar_level)

def predict_health_metrics(user_features):
    prediction_systolic_bp = model_systolic_bp.predict([user_features])
    prediction_diastolic_bp = model_diastolic_bp.predict([user_features])
    prediction_sugar_level = model_sugar_level.predict([user_features])
    return {
        'systolic_bp': round(prediction_systolic_bp[0]),
        'diastolic_bp': round(prediction_diastolic_bp[0]),
        'sugar_level': round(prediction_sugar_level[0])
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get user input from form
            user_features = [
                float(request.form['current_sugar']),
                float(request.form['Csystolic_bp']),
                float(request.form['Cdiastolic_bp']),
                float(request.form['previous_sugar']),
                float(request.form['Psystolic_bp']),
                float(request.form['Pdiastolic_bp'])
            ]
            
            # Predict metrics
            results = predict_health_metrics(user_features)
            
            # Pass the results and user features to the results page
            return render_template('results.html', results=results, user_features=user_features)
        except ValueError:
            return render_template('index.html', error="Invalid input. Please enter numeric values.")
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
