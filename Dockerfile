FROM tiangolo/meinheld-gunicorn-flask:python3.8
ADD *.py /code/
ADD requirements.txt /code/
WORKDIR /code
RUN pip install -r requirements.txt
EXPOSE 8888
CMD ["gunicorn"  , "-b", "0.0.0.0:8888", "app:server", "--workers", "4"]