FROM ubuntu:16.04
MAINTAINER Diana Sousa (dfsousa@lasige.di.fc.ul.pt)


WORKDIR /bolstm


# --------------------------------------------------------------
#                         GENERAL SET UP
# --------------------------------------------------------------

RUN apt-get update -y && apt-get install wget -y


# --------------------------------------------------------------
#                  CREATE/COPY REPOSITORY DIRECTORIES                
# --------------------------------------------------------------

COPY src/ src/
RUN mkdir /bolstm/temp/
RUN mkdir /bolstm/models/
RUN mkdir /bolstm/data/

# --------------------------------------------------------------
#               PYTHON LIBRARIES AND CONFIGURATION
# --------------------------------------------------------------

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev && apt-get autoclean -y
#RUN apt-get update && apt-get install sqlite3 libsqlite3-dev -y
RUN ln -s $(which pip3) /usr/bin/pip
RUN pip install --upgrade pip
RUN pip install numpy==1.13.3
RUN pip install tensorflow==1.5.0
RUN pip install gensim==3.1.0
RUN pip install Keras
RUN pip install sklearn==0.0
RUN pip install matplotlib
RUN apt-get update && apt-get install -y git && apt-get autoclean -y
RUN git clone https://github.com/dpavot/obonet
RUN cd obonet && python3 setup.py install
RUN pip install fuzzywuzzy==0.15.1
RUN pip install spacy==2.0.10
RUN pip install scipy==1.0.0
RUN pip install python-Levenshtein==0.12.0
RUN python3 -m spacy download en_core_web_sm


# --------------------------------------------------------------
#                GENIASS (REQUIRES RUBY AND MAKE)
# --------------------------------------------------------------

WORKDIR /bolstm/bin
RUN wget -q http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && \
    tar -xvzf geniass-1.00.tar.gz && \
    rm geniass-1.00.tar.gz  
WORKDIR /bolstm/bin/geniass/
RUN apt-get update -y && apt-get install -y build-essential g++ make && make


# --------------------------------------------------------------
#                         SST LIGHT 0.4
# --------------------------------------------------------------

WORKDIR /bolstm
#RUN wget https://sourceforge.net/projects/supersensetag/files/sst-light/sst-light-0.4/sst-light-0.4.tar.gz && \
#    tar -xvzf sst-light-0.4.tar.gz && \
#    rm sst-light-0.4.tar.gz
RUN git clone https://github.com/AndreLamurias/sst-light.git
RUN mv sst-light sst-light-0.4
WORKDIR /bolstm/sst-light-0.4/
RUN apt-get update -y && make # (error to solve)
RUN cp sst /bin/

# --------------------------------------------------------------
#                             DiShIn
# --------------------------------------------------------------

WORKDIR /bolstm/src
RUN git clone https://github.com/lasigeBioTM/DiShIn.git
WORKDIR /bolstm/src/DiShIn
RUN wget -q www.geneontology.org/ontology/go.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/go.db
RUN wget -q http://purl.obolibrary.org/obo/hp.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/hp.db
RUN wget -q ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/chebi.db

WORKDIR /bolstm/data
RUN wget -q ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo
RUN wget -q http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin


WORKDIR /bolstm
