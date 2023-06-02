import pandas as pd
import pymysql
from sqlalchemy import create_engine

# 讀取CSV檔案並去除欄位空白
df = pd.read_csv("merged.csv")
df.columns = df.columns.str.strip()

# 資料庫連線設定
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'airdb'
}

# 建立資料庫連線
conn = pymysql.connect(**db_config)
engine = create_engine(f'mysql+pymysql://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}/{db_config["database"]}')

# 將資料保存到MySQL資料表
df.to_sql('taiwan', con=engine, if_exists='replace', index=False)

# 關閉資料庫連線
conn.close()

print("DataFrame已保存到MySQL表中。")
