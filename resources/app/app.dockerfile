# vim: set ft=dockerfile :


FROM ghcr.io/prefix-dev/pixi:0.45.0
LABEL maintainer="Mike Anselmi <git@manselmi.com>"

# Change working directory.
WORKDIR /app

# Install pixi requirements.
COPY ["pyproject.toml", "."]
COPY ["pixi.lock", "."]
RUN ["pixi", "install", "--frozen"]
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"
