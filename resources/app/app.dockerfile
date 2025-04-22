# vim: set ft=dockerfile :


FROM ghcr.io/prefix-dev/pixi:0.45.0 AS build
WORKDIR /app
COPY ["pyproject.toml", "."]
COPY ["pixi.lock", "."]
RUN ["pixi", "install", "--frozen"]

FROM gcr.io/distroless/base-debian12:latest
WORKDIR /app
COPY --from=build ["/app/.pixi/envs/default", "/app/.pixi/envs/default"]
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"

LABEL maintainer="Mike Anselmi <git@manselmi.com>"
