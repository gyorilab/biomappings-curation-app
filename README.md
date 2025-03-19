<!-- vim: set ft=markdown : -->


# Biomappings curation app

``` shell
caddy run --config Caddyfile
oauth2-proxy --config <(op inject -i oauth2-proxy.toml)
docker container run --rm -it -p 8080:80 docker.io/kennethreitz/httpbin:latest
```

[Biomappings curation app](https://app.gyori-mac.localdomain)
