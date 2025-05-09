# vim: set ft=dockerfile :


ARG CADDY_EXTRA_PLACEHOLDERS_VERSION=${CADDY_EXTRA_PLACEHOLDERS_VERSION:-main}
ARG CADDY_VERSION=${CADDY_VERSION:-latest}

FROM docker.io/library/caddy:${CADDY_VERSION}-builder AS builder
RUN xcaddy build \
      --with="github.com/steffenbusch/caddy-extra-placeholders@${CADDY_EXTRA_PLACEHOLDERS_VERSION}"

FROM docker.io/library/caddy:${CADDY_VERSION}
COPY --from=builder ["/usr/bin/caddy", "/usr/bin/caddy"]
