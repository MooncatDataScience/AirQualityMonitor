import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os



t_current_dir = os.path.dirname(os.path.abspath(__file__))
t_model_dir = os.path.join(t_current_dir, '../templates')

app = FastAPI()
templates = Jinja2Templates(directory=t_model_dir)


# 定義輸入數據格式
class InputData(BaseModel):
    model_name: str


# 載入模型



import os

MODELS = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../models')

for filename in os.listdir(model_dir):
    if filename.endswith('.pkl'):
        model_path = os.path.join(model_dir, filename)
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        model_name = os.path.splitext(filename)[0]
        MODELS[model_name] = model


def feature_selection(table):
    ds = table[['pm2.5_avg', 'pm10_avg']]
    status = table['aqi']
    return ds, status


def predict_aqi(model, data):
    data.fillna(0, inplace=True)
    X, y = feature_selection(data)
    n = X[['pm2.5_avg', 'pm10_avg']]
    n = n.to_numpy()
    y_pred = model.predict(n)
    return dict(enumerate(y_pred.flatten(), 1))

sites = ['沙鹿', '豐原', '大里', '忠明', '西屯']


# 定義首頁
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    context = {"request": request, "title": "Home"}
    return templates.TemplateResponse("home.html", context=context)


# 定義 API 路由
@app.post("/{model_name}")
async def predict(model_name: str):
    model = MODELS.get(model_name)
    if not model:
        return JSONResponse(content={"message": "Model not found."}, status_code=404)
    
    results = []
    for site in sites:
        df = pd.read_csv("aqi.csv")
        df['publishtime'] = pd.to_datetime(df['publishtime'])
        df['hour'] = df['publishtime'].apply(lambda x: x.hour)
        table = df[df['site'] == site]
        site_results = predict_aqi(model, table)
        site_results['Site'] = site
        results.append(site_results[len(site_results)-1])
    
    dic = {
        '模型' : str(model_name),
        '沙鹿' : results[0],
        '豐原' : results[1],
        '大里' : results[2],
        '忠明' : results[3],
        '西屯' : results[4]
    }
    
    return dic
   
        

    
