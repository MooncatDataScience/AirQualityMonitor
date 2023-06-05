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

# 資料庫連線設定
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'airdb'
}

def connect_to_database():
    return pymysql.connect(**db_config)

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
    data['date'] = pd.to_datetime(df[1])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['substance'] = df[2]

    hour_columns = [str(i) + 'hour' for i in range(24)]
    for i, column in enumerate(hour_columns, start=3):
        data[column] = pd.to_numeric(df[i], errors='coerce')

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

def main():
    conn = connect_to_database()
    observatory = input("===輸入測站名稱===")
    df = query_by_observatory(conn, observatory)
    conn.close()

    cleaned_data = clean_data(df)
    daily_average = calculate_daily_average(cleaned_data)

    x, y = feature_selection(daily_average)

    X_train, X_test, y_train, y_test = split(x, y)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)


    model_name = input("===選擇模型===")
    model = get_model(model_name) 
    
    train_model(model, X_train, y_train, X_test, y_test)

    prediction = model.predict(X_test)
    print(prediction)

    #======模型驗證======
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape )
    print(prediction.shape)

    print(f'MSE : {mean_squared_error(prediction, y_test)}')
    print(f'RMSE : {mean_squared_error(prediction, y_test, squared=False)}')
    print(f'MSE : {mean_absolute_error(prediction, y_test)}')
    print(f'R2 : {r2_score(prediction, y_test)}')

if __name__ == '__main__':
    main()
