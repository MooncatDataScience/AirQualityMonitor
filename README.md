# AirQualityMonitor

### 日期：2023年6月7日

### 紀錄人：Mooncat

## 專案說明

### ⚠️ 記得先去雲端下載SQL

| 專案名稱 | 說明 | 備註 |
| -------- | -------- | -------- |
| app     | 主專案部屬     |     |
| dockerfile     |  部屬環境    |     
| requirements.txt| fastapi相依 |     
| package.txt | ML相依套件 |     |


## 系統環境
* docker

* python 3.9

* MySql 8.0

* Ubuntu 22.04

## 開發環境

* vscode

## 安裝手冊
![image](https://github.com/MooncatDataScience/AirQualityMonitor/assets/48311280/66dc7f68-45b3-44cb-87d0-86985ef8785e)

### 參考hackmd
>https://hackmd.io/nvO7TAHFSxyypwS64EiCqg


![image](https://github.com/MooncatDataScience/AirQualityMonitor/assets/48311280/5df2e07c-07f0-417e-acc3-38f91729c2bc)

### 參考hackmd
>https://hackmd.io/nvO7TAHFSxyypwS64EiCqg


![image](https://github.com/MooncatDataScience/AirQualityMonitor/assets/48311280/2d5e9344-b977-430a-904c-c1468f22ef5e)

![](https://i.imgur.com/pK9nYiu.png)

1. 安裝MySQL
2. `mysql -u username -p` 
3. run -d -p 執行容器
4. exeit 進入容器 
5. ls 確認database 位子
6. 執行以下SQL指令
```
CREATE DATABASE database_name;
USE database_name;
SOURCE /path/to/file.sql;
```
`var/airdb.sql;`

* fcitx 中文輸入法
> https://ivonblog.com/posts/ubuntu-fcitx5/

#### 解決不能輸入中文問題
```
sudo docker exec -it <id> env LANG=C.UTF-8  /bin/bash
```
7. 查詢指令 `SELECT * FROM taiwan WHERE TRIM(測站) = '斗六';`
(MySQL到此end)

---
### fastapi docker 請看官方安裝即可
* 記得要同步主機
`uvicorn main:app --reload`
---
### 容器溝通
> https://ithelp.ithome.com.tw/articles/10242460
```
sudo docker network create mynetwork
sudo docker network connect 容器1
sudo docker network connect 容器2
```

## 其他說明
* fastapi docker 要先能ping 到 MySQL docker

* 容器要先安裝合適套件
`apt-get update && apt-get install -y iputils-ping`

* 測試ping
```
docker exec -it [容器名稱或容器ID] ping [目標IP或主機名稱]
```

## 技術總結

1. fastapi 串接 MySQL 架設在docker上
2. pymysql的應用
3. 容器的溝通
4. 機器學習的部屬
