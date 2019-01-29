FROM python:3.7

# Dependencies
RUN apt-get update -y && apt-get install -y \
    wget \
    build-essential \
    gcc \
    zlib1g-dev \
    apt-utils \
    mecab \
    libmecab-dev \
    make \
    curl \
    git \
    xz-utils \
    file \
    swig \
    sudo \
    software-properties-common \
    locales \
    tar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Neologd installation
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && git checkout -b v0.0.6 \
    && bin/install-mecab-ipadic-neologd -n -y

# Clean up
RUN rm -rf /mecab-ipadic-neologd

# Enable Japanese
RUN locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

RUN python3.7 -m pip install pip --upgrade

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
