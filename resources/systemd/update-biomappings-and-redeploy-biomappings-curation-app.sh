#!/usr/bin/env bash
# vim: set ft=bash :

# Stop at any error, treat unset vars as errors and make pipelines exit with a non-zero exit code if
# any command in the pipeline exits with a non-zero exit code.
set -o errexit
set -o nounset
set -o pipefail


msg() {
  printf '%s\n' "${1}" >&2
}


APP_SERVICE="${PROJECT_NAME}.service"
REPO_DIR=resources/biomappings


if ! systemctl --quiet is-active -- "${APP_SERVICE}"; then
  msg "${APP_SERVICE} is not active - skipping update check"
  exit 0
fi

pixi run -- fetch-biomappings-repo
LOCAL_REV="$(git -C "${REPO_DIR}" rev-parse --verify --end-of-options HEAD)"
UPSTREAM_REV="$(git -C "${REPO_DIR}" rev-parse --verify --end-of-options HEAD@{upstream})"

if [[ "${LOCAL_REV}" == "${UPSTREAM_REV}" ]]; then
  msg 'local branch is up-to-date with upstream branch - skipping redeploy'
  exit 0
fi

sudo -- systemctl stop -- "${APP_SERVICE}"
git -C "${REPO_DIR}" reset --hard HEAD@{upstream}
docker volume rm -f -- "${PROJECT_NAME}_postgres-data"
sudo -- systemctl start -- "${APP_SERVICE}"
