<!-- vim: set ft=markdown : -->


# Biomappings curation app

## Update `/etc/hosts`

``` shell
printf -- '%s\n' \
    '::1 app.gyori-mac.localdomain' \
    '127.0.0.1 app.gyori-mac.localdomain' \
  | sudo -- tee -a -- /etc/hosts > /dev/null
```

## TLS cert

Place a trusted TLS cert with SAN `*.gyori-mac.localdomain` and corresponding key at the locations
matching those of the `tls` directive within `Caddyfile`.

## Launch

``` shell
pixi run -- caddy
pixi run -- oauth2-proxy
pixi run -- flask
```

[Biomappings curation app](https://app.gyori-mac.localdomain)
