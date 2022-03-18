FROM python:3.7

CMD ["/bin/bash"]

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
   &&  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

#RUN apt-get update \
#   &&  apt-get install -y --no-install-recommends libmlx4-1 libmlx5-1 librdmacm1 libibverbs1 libmthca1 libdapl2 dapl2-utils openssh-client openssh-server iproute2 \
#   &&  apt-get install -y build-essential bzip2=1.0.6-8ubuntu0.2 libbz2-1.0=1.0.6-8ubuntu0.2 systemd git wget cpio libsm6 libxext6 libxrender-dev fuse \
#   &&  apt-get clean -y \
#   &&  rm -rf /var/lib/apt/lists/*
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
   &&  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 5001 8883 8888 9000
EXPOSE 25300-25600
USER root:root
WORKDIR /opt
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
RUN apt-get update \
   &&  apt-get install -y --no-install-recommends redis-server default-jre
RUN apt-get install -y --no-install-recommends unzip
RUN  unzip stanford-corenlp-full-2018-10-05.zip \
   &&  mv $(ls -d stanford-corenlp-full-*/) corenlp \
   &&  rm *.zip
EXPOSE 5002-5100
EXPOSE 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011 5012 5013 5014 5015 5016 5017 5018
EXPOSE 50022 50023 50024 50025 50026 50027 50028 50029 50030 50031 50032 50034 50035 50036 50037 50038 50039
EXPOSE 50022 50032 50042 50052 50062 50072 50082 50092 50102 50112 50122 50132 50142 50152 50162 50172 50182
COPY . /tdqn-scienceworld
RUN pip install -r /tdqn-scienceworld/requirements.txt


RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python -m spacy download en

WORKDIR /

ENV PYTHONPATH=/tdqn-scienceworld/drrn
ENV HOME=""
WORKDIR /tdqn-scienceworld/drrn