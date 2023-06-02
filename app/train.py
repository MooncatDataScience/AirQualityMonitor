from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os

def feature_selection(table):
    ds = table[['pm2.5_avg', 'pm10_avg']]
    status = table['aqi']
    return ds, status

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train.astype(float), X_test.astype(float), y_train.astype(float), y_test.astype(float)

def get_model(model_name):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machine": SVR(),
        "Neural Network": MLPRegressor(),
        "XGBoost": xgb.XGBRegressor()
    }
    return models[model_name]


def train_model(model, X_train, y_train, X_test, y_test):
    if isinstance(model, xgb.XGBRegressor):
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    else:
        model.fit(X_train, y_train)


def save_model(model, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_path = os.path.join(model_dir, model_name)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")


def predict_model(X_train, y_train, X_test, y_test, model_name, model_dir='models', model_file='my_model'):
    model = get_model(model_name)
    train_model(model, X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    y_pred = ['%.1f' % elem for elem in y_pred]
    y_test = ['%.1f' % elem for elem in y_test]
    tail = str(y_pred[-1]).replace("[", "").replace("]", "").replace("'", "")
    current = str(y_test[-1]).replace("[", "").replace("]", "").replace("'", "")
    #save_model(model, model_dir, model_file + '.pkl')
    return {'MAE': round(mae, 2), 'MSE': round(mse, 2), 'R2': round(r2, 2), 'Current': current, 'Prediction': tail}



#     return {'Model': model_file + '.pkl', 'MAE': round(mae, 2), 'MSE': round(mse, 2), 'R2': round(r2, 2), 'Current': current, 'Prediction': tail}
