FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git

# Install pyenv
RUN curl https://pyenv.run | bash

# Set up pyenv environment variables
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install multiple Python versions
RUN pyenv install 3.8.12
RUN pyenv install 3.9.7
RUN pyenv install 3.10.2

WORKDIR /app

# Install Python dependencies for each version and upgrade pip
RUN pyenv global 3.8.12 && /root/.pyenv/shims/python -m pip install --upgrade pip numpy flake8 pytest
RUN pyenv global 3.9.7 && /root/.pyenv/shims/python -m pip install --upgrade pip numpy flake8 pytest
RUN pyenv global 3.10.2 && /root/.pyenv/shims/python -m pip install --upgrade pip numpy flake8 pytest
