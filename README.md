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

* Clone the Biomappings repository:

    ``` shell
    pixi run -- clone-biomappings-repo
    ```


## Update `/etc/hosts`

``` shell
printf -- '%s\n' \
    '::1 curate.biomappings.localdomain' \
    '127.0.0.1 curate.biomappings.localdomain' \
  | sudo -- tee -a -- /etc/hosts > /dev/null
```

## Start

``` shell
pixi run -- up
```

## Browse local app

[Biomappings curation app](https://curate.biomappings.localdomain)

## Stop

``` shell
pixi run -- down
```

## Deploy

First, ensure you have the deployment host configured as a SSH destination named
`biomappings-curation-app` in `~/.ssh/config`. I would suggest enabling SSH connection multiplexing
in a manner similar to the following, as the deployment process runs multiple SSH commands against
the deployment host:

``` text
Host biomappings-curation-app
  …

Host *
  ControlMaster auto
  ControlPath ~/.ssh/sockets/%C
  ControlPersist 30s
```

Commit all changes you'd like to deploy, then run:

``` shell
pixi run -- deploy
```

## Browse deployed app

[Biomappings curation app](https://biomappings-curation-app-lb-00cc5d7d789bc0c6.elb.us-east-1.amazonaws.com)
