# vim: set ft=dockerfile :


ARG PIXI_VERSION=${PIXI_VERSION:-latest}

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}
WORKDIR /app
COPY ["pyproject.toml", "."]
COPY ["pixi.lock", "."]
RUN pixi install --frozen && rm -fr -- "${HOME}/.cache/rattler"
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"
