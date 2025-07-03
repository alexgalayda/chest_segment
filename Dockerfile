ARG BUILD_TARGET=cpu
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS gpu-base
FROM ubuntu:24.04 AS cpu-base

FROM ${BUILD_TARGET}-base

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV TZ=Asia/Yerevan
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/home/${USER_NAME}/.local/bin:${PATH}"

RUN userdel -r ubuntu

RUN groupadd -f -g $GROUP_ID $USER_NAME
RUN useradd -u $USER_ID -g $GROUP_ID -m $USER_NAME

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update --fix-missing && \
    apt-get upgrade -y

RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_HTTP_TIMEOUT=12000

RUN mkdir -p ${UV_CACHE_DIR}
RUN chown -R ${USER_NAME}:${USER_NAME} ${UV_CACHE_DIR}

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/chest_segment

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv python install 3.12
RUN uv python pin 3.12

RUN --mount=type=cache,target=/var/cache/uv,uid=${USER_ID},gid=${GROUP_ID} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    UV_CACHE_DIR=${UV_CACHE_DIR} UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT} uv sync --no-install-project

ADD . /home/${USER_NAME}/chest_segment

RUN --mount=type=cache,target=/var/cache/uv,uid=${USER_ID},gid=${GROUP_ID} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    UV_CACHE_DIR=${UV_CACHE_DIR} UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT} uv sync

CMD ["uv", "run", "run.py"]
