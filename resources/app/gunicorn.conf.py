# https://github.com/benoitc/gunicorn/issues/1352
# https://gist.github.com/hynek/ba655c8756924a5febc5285c712a7946

import os


def on_starting(server):
    """
    Attach a set of IDs that can be temporarily re-used.

    Used on reloads when each worker exists twice.
    """
    server._worker_id_overload = set()


def nworkers_changed(server, new_value, old_value):  # noqa: ARG001
    """
    Gets called on startup too.

    Set the current number of workers. Required if we raise the worker count temporarily using TTIN
    because server.cfg.workers won't be updated and if one of those workers dies, we wouldn't know
    the ids go that far.
    """
    server._worker_id_current_workers = new_value


def _next_worker_id(server):
    """
    If there are IDs open for re-use, take one.  Else look for a free one.
    """
    if server._worker_id_overload:
        return server._worker_id_overload.pop()

    in_use = {w._worker_id for w in tuple(server.WORKERS.values()) if w.alive}
    free = set(range(server._worker_id_current_workers)) - in_use

    return free.pop()


def on_reload(server):
    """
    Add a full set of ids into overload so it can be re-used once.
    """
    server._worker_id_overload = set(range(server.cfg.workers))


def pre_fork(server, worker):
    """
    Attach the next free worker_id before forking off.
    """
    worker._worker_id = _next_worker_id(server)


def post_fork(server, worker):  # noqa: ARG001
    """
    Put the worker_id into an env variable for further use within the app.
    """
    os.environ["APP_WORKER_ID"] = str(worker._worker_id)
