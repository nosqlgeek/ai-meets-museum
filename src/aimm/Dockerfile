FROM python:3.9
WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install --upgrade redis

RUN mkdir ImgUpload
RUN mkdir static/ImgStore
RUN mv entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["python", "app.py"]