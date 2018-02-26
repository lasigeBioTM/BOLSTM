FROM python:3.5
MAINTAINER Andre Lamurias (alamurias@lasige.di.fc.ul.pt)

RUN apt-get update && apt-get install -y pandoc && apt-get autoclean -y
RUN apt-get update && apt-get install -y git && apt-get autoclean -y
RUN apt-get update && apt-get install -y wget && apt-get autoclean -y
RUN apt-get update && apt-get install -y default-jre && apt-get autoclean -y

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_lg

RUN git clone https://github.com/AndreLamurias/obonet.git
RUN cd obonet && python3 setup.py install


RUN mkdir ./temp

COPY ./sst-light-0.4 ./sst-light-0.4
RUN cd sst-light-0.4 && make
ENV PATH="/sst-light-0.4:$PATH"
RUN pip3 install stanford-corenlp
RUN echo "deb http://http.debian.net/debian jessie-backports main" | \
      tee --append /etc/apt/sources.list.d/jessie-backports.list > /dev/null
RUN apt-get update && apt-get install -y -t jessie-backports openjdk-8-jdk
RUN update-java-alternatives -s java-1.8.0-openjdk-amd64


COPY evaluateDDI.jar ./
