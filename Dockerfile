FROM python:3.9
LABEL authors="arthur"

RUN apt-get update

RUN apt-get install -y gdal-bin=3.6.2+dfsg-1+b2
RUN apt-get install -y libgdal-dev

RUN apt-get install -y python3-pip

RUN pip install --upgrade pip

RUN mkdir /home/planet-spy-report

WORKDIR /home/planet-spy-report

COPY . ./

RUN pip install -r requirements.txt

RUN mkdir /home/planet-spy-report/volume
VOLUME /home/planet-spy-report/volume

CMD bash