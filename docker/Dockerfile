
FROM python:3.7.6

RUN apt-get update -y && apt-get upgrade -y

COPY ./docker/requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /usr/src
