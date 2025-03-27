<!-- vim: set ft=markdown : -->


# Biomappings curation app

## Update `/etc/hosts`

``` shell
printf -- '%s\n' \
    '::1 app.gyori-mac.localdomain' \
    '127.0.0.1 app.gyori-mac.localdomain' \
  | sudo -- tee -a -- /etc/hosts > /dev/null
```

## Inject secrets from credential store

``` shell
find -- ./env -type f -name '*.secret.tpl' \
  -exec zsh -efuc 'for ARG; do op inject -f -i "${ARG}" -o "${ARG:r}"; done' zsh {} +
```

## TLS cert

Place a trusted TLS cert with SAN `*.gyori-mac.localdomain` and corresponding key at
`~/.pki/private/x509/gyori-mac/{cert.pem,key.pem}`, respectively.

## Launch

``` shell
docker-compose up
```

[Biomappings curation app](https://app.gyori-mac.localdomain)
