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
    '::1 curate.biomappings.io' \
    '127.0.0.1 curate.biomappings.io' \
  | sudo -- tee -a -- /etc/hosts > /dev/null
```

## Launch

``` shell
pixi run -- up
```

[Biomappings curation app](https://curate.biomappings.io)
