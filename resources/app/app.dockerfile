# vim: set ft=dockerfile :


FROM ghcr.io/prefix-dev/pixi:0.44.0
LABEL maintainer="Mike Anselmi <git@manselmi.com>"

# Change working directory.
WORKDIR /app

# Install pixi requirements.
COPY ["pyproject.toml", "."]
COPY ["pixi.lock", "."]
RUN ["pixi", "install", "--frozen"]

# Configure default behavior.
COPY ["resources/app/app.entrypoint", "/entrypoint"]
ENTRYPOINT ["/entrypoint"]
EXPOSE 5000/tcp
