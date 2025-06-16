# vim: set ft=dockerfile :


ARG PIXI_VERSION=${PIXI_VERSION:-latest}

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}
WORKDIR /app
COPY ["pyproject.toml", "."]
COPY ["pixi.lock", "."]
# Git is needed because Pixi is installing dependencies (e.g. biomappings) via Git.
RUN apt update \
    && apt -y install --no-install-recommends -- ca-certificates git \
    && apt clean all \
    && rm -fr -- /var/lib/apt/lists/*
RUN pixi install --frozen && rm -fr -- "${HOME}/.cache/rattler"
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"
