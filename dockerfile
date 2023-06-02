FROM python:3.9

# # 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app/ /code/app/

COPY models/ /code/models/

# 复制templates目录到容器中
COPY templates/ /code/templates/

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
