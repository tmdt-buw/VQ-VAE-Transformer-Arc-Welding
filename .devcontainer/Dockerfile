FROM mambaorg/micromamba:1.4.4-focal-cuda-11.7.1
# ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --from=tianon/gosu /usr/local/bin/gosu /usr/local/bin/gosu
USER root

ENV USERNAME=mambauser
# RUN pip install future-fstrings
RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -y git
RUN apt update
RUN apt install git -y 

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN --mount=type=cache,mode=0777,uid=1000,gid=1000,target=/opt/conda/pkgs micromamba install -y -n base -f /tmp/env.yaml

USER $MAMBA_USER

EXPOSE 8888
EXPOSE 8000

WORKDIR /home/mambauser/dev-container
