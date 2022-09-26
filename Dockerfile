FROM nvcr.io/nvidia/pytorch:22.03-py3

EXPOSE 8888

VOLUME ["/sources"]

WORKDIR /sources

RUN pip install --upgrade pip \
    && pip3 install torchinfo \
    && pip3 install fastBPE \
    && pip3 install dtw-python \
    && pip3 install nltk \
    && pip install keras \
    && pip3 install tqdm \
    && pip3 install matplotlib \
    && pip3 install pretty_midi \
    && pip3 install PyYAML \
    && pip3 install transformers==3.5.1 \
    && pip3 install pypinyin==0.39.1 \
    && pip3 install jieba==0.42.1 \
    && pip3 install tensorboard==2.4.0 \
    && pip3 install librosa \
    && pip3 install pyworld \
    && pip3 install soundfile \
    && pip3 install pyOpenSSL \
    && pip3 install transformers==3.5.1 \
    && pip install tensorflow \
    && pip install ipywidgets


WORKDIR /sources

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--port", "8000"]
