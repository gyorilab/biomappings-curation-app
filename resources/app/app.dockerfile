# vim: set ft=dockerfile :


FROM ghcr.io/prefix-dev/pixi:0.46.0
WORKDIR /app
COPY ["pyproject.toml", "."]
COPY ["pixi.lock", "."]
RUN pixi install --frozen && rm -fr -- "${HOME}/.cache/rattler"
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"

LABEL maintainer="Mike Anselmi <git@manselmi.com>"
