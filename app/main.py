import pymysql
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn import preprocessing
from fastapi import FastAPI

app = FastAPI(debug=True)

# 資料庫連線設定

def connect_to_database():
    connection = pymysql.connect(
        host='172.17.0.3',
        user='root',
        password='123456',
        database='airdb',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

def query_by_observatory(conn, observatory):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM taiwan WHERE trim(測站) = %s"
            cursor.execute(sql, observatory)
            result = cursor.fetchall()

            if result:
                return pd.DataFrame(result)
            else:
                print("找不到資料")
    except pymysql.Error as e:
        print(f"查詢發生錯誤：{e}")

def clean_data(df):
    data = pd.DataFrame()
    data['date'] = pd.to_datetime(df['日期'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['substance'] = df['測項']

    hour_columns = [str(i) + 'hour' for i in range(24)]
    for i, column in enumerate(hour_columns):
        x = '0'+str(i)
        if i < 10:
            data[column] = pd.to_numeric(df[x], errors='coerce')
        else:
            data[column] = pd.to_numeric(df[str(i)], errors='coerce')

    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return data

def calculate_daily_average(data):
    hour_columns = [str(i) + 'hour' for i in range(24)]
    data[hour_columns] = data[hour_columns]
    data['daily_average'] = data[hour_columns].mean(axis=1)
    return data

def feature_selection(data):
    data['x1'] = data['daily_average'] * 1.5
    data['x2'] = data['daily_average'] * 3.14
    x = data[['x1', 'x2']]
    y = data['daily_average']
    return x, y

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


@app.get("/")
async def root():
    return {"message": "Hello World TEST14"}


@app.post("/forecast/{observatory}/{model_name}")
async def make_prediction(observatory: str, model_name: str):
    conn = connect_to_database()
    df = query_by_observatory(conn, observatory)
    if df is None:
        return {"error": "找不到資料"}

    try:
        cleaned_data = clean_data(df)
        averaged_data = calculate_daily_average(cleaned_data)
        selected_features, target = feature_selection(averaged_data)
        X_train, X_test, y_train, y_test = split(selected_features, target)
        model = get_model(model_name)
        train_model(model, X_train, y_train, X_test, y_test)
        prediction = model.predict(X_test)
        return {"observatory": observatory, "model": model_name, "prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}
 


