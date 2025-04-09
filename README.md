<!-- vim: set ft=markdown : -->


# Biomappings curation app

## Prerequisites

* [Pixi](https://pixi.sh)

* Application capable of building and running Docker Compose stacks, such as
  [OrbStack](https://orbstack.dev)

* Configure Git hooks:

    ``` shell
    pixi run -- pre-commit-install
    ```

* Configure Git diff behavior for encrypted files:

    ``` shell
    pixi run -- configure-sops-diff
    ```

  Please see [SOPS: Showing diffs in cleartext in
  Git](https://github.com/getsops/sops#showing-diffs-in-cleartext-in-git) and
  [`.gitattributes`](.gitattributes) to learn more.

* Update `/etc/hosts`:

    ``` shell
    printf -- '%s\n' \
        '::1 curate.biomappings.localdomain' \
        '127.0.0.1 curate.biomappings.localdomain' \
      | sudo -- tee -a -- /etc/hosts > /dev/null
    ```

## Local development

### Gain access to [`env/secret.env.sops.env`](env/secret.env.sops.env)

You'll need to create an [age](https://github.com/FiloSottile/age#readme)
keypair and write the private key to a [file where SOPS can find
it](https://github.com/getsops/sops#23encrypting-using-age).

On macOS, this file is located at

``` shell
KEYS_FILE="${XDG_CONFIG_HOME:-"${HOME}/Library/Application Support"}/sops/age/keys.txt"
```

On Linux, this file is located at

``` shell
KEYS_FILE="${XDG_CONFIG_HOME:-"${HOME}/.config"}/sops/age/keys.txt"
```

Generate a keypair and append it to `$KEYS_FILE`:

``` shell
mkdir -p -- "$(dirname -- "${KEYS_FILE}")"
touch -- "${KEYS_FILE}"
chmod -- 600 "${KEYS_FILE}"
age-keygen >> "${KEYS_FILE}"
```

Take the public key from `$KEYS_FILE` that you just generated and send it to Mike, who will add it
to [`.sops.yaml`](.sops.yaml) and re-encrypt [`env/secret.env.sops.env`](env/secret.env.sops.env) so
that it may be decrypted in your environment.

### Start Compose stack

First, clone the Biomappings repository with

``` shell
pixi run -- clone-biomappings-repo
```

and then bring up the Compose stack:

``` shell
pixi run -- up
```

### Browse local app

[Biomappings curation app (local)](https://curate.biomappings.localdomain)

### Stop Compose stack

``` shell
pixi run -- down
```

## Deploying changes

First, ensure you have the deployment host configured as a SSH destination named
`biomappings-curation-app` in `~/.ssh/config`. I would suggest enabling SSH connection multiplexing
with `ControlPersist` set to a non-zero timeout, as the deployment process runs multiple SSH
commands against the deployment host. For example:

``` text
Host biomappings-curation-app
  …

Host *
  ControlMaster auto
  ControlPath ~/.ssh/%C
  ControlPersist 30s
```

### Deploy

Commit all changes you'd like to deploy, then run:

``` shell
pixi run -- deploy
```

### Browse deployed app

[Biomappings curation app
(deployed)](https://biomappings-curation-app-lb-00cc5d7d789bc0c6.elb.us-east-1.amazonaws.com)
