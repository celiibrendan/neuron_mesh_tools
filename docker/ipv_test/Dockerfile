FROM ubuntu:18.04
LABEL maintainer="Brendan Papadopoulos"
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel

# duplicated dependencies but previously missing parts are necessary for building python-igraph
RUN apt-get -y install build-essential python3-dev libxml2 libxml2-dev zlib1g-dev

RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip
RUN pip3 install datajoint --upgrade
# RUN pip3 install python-igraph xlrd
RUN apt-get -y install python3-igraph
RUN pip3 install xlrd
RUN apt-get -y install vim
RUN pip3 install matplotlib==3.0.3



WORKDIR /src

RUN pip3 install ipyvolume jupyterlab statsmodels pycircstat nose
RUN pip3 install seaborn --upgrade
RUN pip3 install jgraph


EXPOSE 8888


RUN pip3 install pykdtree trimesh 


#RUN apt-get install -y python3-rtree
#RUN apt-get install -y nodejs npm
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
#RUN jupyter labextension install ipyvolume


WORKDIR /notebooks

ADD ./notebooks .

ADD ./jupyter/run_jupyter_unix.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter_unix.sh"]