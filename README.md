<!-- vim: set ft=markdown : -->


# Biomappings curation app

## Prerequisites

1. [Pixi](https://pixi.sh)

1. Application capable of building and running Docker Compose stacks, such as
   [OrbStack](https://orbstack.dev)

1. Configure Git hooks:

    ``` shell
    pixi run -- pre-commit-install
    ```

1. Configure Git diff behavior for encrypted files:

    ``` shell
    pixi run -- configure-sops-diff
    ```

   Please see [SOPS: Showing diffs in cleartext in
   Git](https://github.com/getsops/sops#showing-diffs-in-cleartext-in-git) and
   [`.gitattributes`](.gitattributes) to learn more.

1. Update `/etc/hosts`:

    ``` shell
    printf -- '%s\n' \
        '::1 curate.biomappings.localdomain' \
        '127.0.0.1 curate.biomappings.localdomain' \
      | sudo -- tee -a -- /etc/hosts > /dev/null
    ```

1. Gain access to [`env/secret.sops.env`](env/secret.sops.env):

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

   Take the public key from `$KEYS_FILE` that you just generated and send it to Mike, who will add
   it to [`.sops.yaml`](.sops.yaml) and re-encrypt [`env/secret.sops.env`](env/secret.sops.env) so
   that it may be decrypted in your environment.

1. Clone the Biomappings repository:

    ``` shell
    pixi run -- clone-biomappings-repo
    ```

1. **[Linux only]** ORCID OAuth2 client configurations support only `https` URLs with the default
   port of 443, so the app must be accessible on port 443.

   This is [not a problem on macOS ≥
   10.14](https://developer.apple.com/forums/thread/674179?answerId=662907022#662907022). Linux may
   be configured such that all ports (including 443) are unprivileged by running:

    ``` shell
    printf -- '%s\n' 'net.ipv4.ip_unprivileged_port_start = 0' \
      | sudo -- tee -a -- /etc/sysctl.d/50-unprivileged-ports.conf > /dev/null
    sudo -- sysctl -q --system
    ```

   Despite the `ipv4` suggesting otherwise, this kernel parameter also [applies to
   IPv6](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=4548b683b78137f8eadeb312b94e20bb0d4a7141).

1. **[Linux only]** If SELinux is enabled and in enforcing mode, make sure that the files to be
   bind-mounted are correctly labeled:

    ``` shell
    chcon -R -t container_file_t -- app.py resources
    ```

## Local development

### Start Compose stack

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
