"""Web curation interface for :mod:`biomappings`."""

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

import bioregistry
import flask
import flask_bootstrap
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass
from sqlalchemy.schema import MetaData, PrimaryKeyConstraint
from werkzeug.local import LocalProxy
from wtforms import StringField, SubmitField

from biomappings.resources import (
    append_false_mappings,
    append_true_mappings,
    append_unsure_mappings,
    load_predictions,
    write_predictions,
)
from biomappings.utils import (
    check_valid_prefix_id,
    commit,
    get_branch,
    get_curie,
    not_main,
    push,
)

RESOURCES_DIR = (
    Path(__file__)
    .resolve()
    .with_name("submodules")
    .joinpath("biomappings")
    .joinpath("src")
    .joinpath("biomappings")
    .joinpath("resources")
)


class SQLAlchemyBase(DeclarativeBase, MappedAsDataclass):
    metadata = MetaData(
        naming_convention={
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "ix": "ix_%(column_0_label)s",
            "pk": "pk_%(table_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
        }
    )


db = SQLAlchemy(model_class=SQLAlchemyBase)


class Marked(db.Model):  # type: ignore[name-defined]
    __tablename__ = "marked"

    user_id: Mapped[str]
    line: Mapped[int]
    value: Mapped[str]

    __table_args__ = (PrimaryKeyConstraint("user_id", "line"),)


class TotalCurated(db.Model):  # type: ignore[name-defined]
    __tablename__ = "total_curated"

    user_id: Mapped[str]
    total_curated: Mapped[int]

    __table_args__ = (PrimaryKeyConstraint("user_id"),)


class AddedMappings(db.Model):  # type: ignore[name-defined]
    __tablename__ = "added_mappings"

    user_id: Mapped[str]
    source_prefix: Mapped[str]
    source_identifier: Mapped[str]
    source_name: Mapped[str]
    relation: Mapped[str]
    target_prefix: Mapped[str]
    target_identifier: Mapped[str]
    target_name: Mapped[str]
    type: Mapped[str]
    prediction_type: Mapped[str | None]
    prediction_source: Mapped[str | None]
    prediction_confidence: Mapped[str | None]

    __table_args__ = (
        PrimaryKeyConstraint(
            "user_id", "source_prefix", "source_identifier", "target_prefix", "target_identifier"
        ),
    )


class TargetIds(db.Model):  # type: ignore[name-defined]
    __tablename__ = "target_ids"

    user_id: Mapped[str]
    prefix: Mapped[str]
    identifier: Mapped[str]

    __table_args__ = (PrimaryKeyConstraint("user_id", "prefix", "identifier"),)


class State(BaseModel):
    """Contains the state for queries to the curation app."""

    limit: int | None = 10
    offset: int | None = 0
    query: str | None = None
    source_query: str | None = None
    source_prefix: str | None = None
    target_query: str | None = None
    target_prefix: str | None = None
    provenance: str | None = None
    prefix: str | None = None
    sort: str | None = None
    same_text: bool | None = None
    show_relations: bool = True
    show_lines: bool = False
    user_id: str

    @classmethod
    def from_flask_globals(cls) -> "State":
        """Get the state from the flask current request."""
        return State(
            limit=flask.request.args.get("limit", type=int, default=10),
            offset=flask.request.args.get("offset", type=int, default=0),
            query=flask.request.args.get("query"),
            source_query=flask.request.args.get("source_query"),
            source_prefix=flask.request.args.get("source_prefix"),
            target_query=flask.request.args.get("target_query"),
            target_prefix=flask.request.args.get("target_prefix"),
            provenance=flask.request.args.get("provenance"),
            prefix=flask.request.args.get("prefix"),
            sort=flask.request.args.get("sort"),
            same_text=_get_bool_arg("same_text"),
            show_relations=_get_bool_arg("show_relations") or current_app.config["SHOW_RELATIONS"],
            show_lines=_get_bool_arg("show_lines") or current_app.config["SHOW_LINES"],
            user_id=f"orcid:{flask.request.headers['X-Auth-Request-User']}",
        )


def _get_bool_arg(name: str, default: bool | None = None) -> bool | None:
    value = flask.request.args.get(name)
    if value is not None:
        return value.lower() in {"true", "t"}
    return default


def url_for_state(endpoint, state: State, **kwargs) -> str:
    """Get the URL for an endpoint based on the state class."""
    vv = state.dict(
        exclude_none=True,
        exclude_defaults=True,
        exclude={
            "user_id",
        },
    )
    vv.update(kwargs)  # make sure stuff explicitly set overrides state
    return flask.url_for(endpoint, **vv)


def get_app(
    predictions_path: Path | None = None,
    positives_path: Path | None = None,
    negatives_path: Path | None = None,
    unsure_path: Path | None = None,
) -> flask.Flask:
    """Get a curation flask app."""
    app_ = flask.Flask(__name__)
    app_.config["WTF_CSRF_ENABLED"] = False
    app_.config["SECRET_KEY"] = os.urandom(8)
    app_.config["SHOW_RELATIONS"] = True
    app_.config["SHOW_LINES"] = False
    controller = Controller(
        predictions_path=predictions_path,
        positives_path=positives_path,
        negatives_path=negatives_path,
        unsure_path=unsure_path,
    )
    if not controller._predictions and predictions_path is not None:
        msg = f"There are no predictions to curate in {predictions_path}"
        raise RuntimeError(msg)
    app_.config["controller"] = controller
    flask_bootstrap.Bootstrap4(app_)
    app_.register_blueprint(blueprint)
    app_.jinja_env.globals.update(
        controller=controller,
        url_for_state=url_for_state,
    )
    app_.config["SQLALCHEMY_DATABASE_URI"] = "".join(
        [
            "postgresql+psycopg://",
            quote(os.environ["SQLALCHEMY_DATABASE_USERNAME"], safe=""),
            ":",
            quote(os.environ["SQLALCHEMY_DATABASE_PASSWORD"], safe=""),
            "@",
            quote(os.environ["SQLALCHEMY_DATABASE_HOSTNAME"], safe=""),
            ":",
            os.environ["SQLALCHEMY_DATABASE_PORT"],
            "/",
            quote(os.environ["SQLALCHEMY_DATABASE_NAME"], safe=""),
        ]
    )
    db.init_app(app_)
    with app_.app_context():
        db.create_all()
    return app_


class Controller:
    """A module for interacting with the predictions and mappings."""

    def __init__(
        self,
        *,
        predictions_path: Path | None = None,
        positives_path: Path | None = None,
        negatives_path: Path | None = None,
        unsure_path: Path | None = None,
    ):
        """Instantiate the web controller.

        :param predictions_path: A custom predictions file to curate from
        :param positives_path: A custom positives file to curate to
        :param negatives_path: A custom negatives file to curate to
        :param unsure_path: A custom unsure file to curate to
        """
        self.predictions_path = predictions_path
        self._predictions = load_predictions(path=self.predictions_path)

        self.positives_path = positives_path
        self.negatives_path = negatives_path
        self.unsure_path = unsure_path

        self._marked: dict[int, str] = {}
        self.total_curated = 0
        self._added_mappings: list[dict[str, None | str | float]] = []
        self.target_ids: set[tuple[str, str]] = set()

    def predictions_from_state(self, state: State) -> Iterable[tuple[int, Mapping[str, Any]]]:
        """Iterate over predictions from a state instance."""
        return self.predictions(
            offset=state.offset,
            limit=state.limit,
            query=state.query,
            source_query=state.source_query,
            source_prefix=state.source_prefix,
            target_query=state.target_query,
            target_prefix=state.target_prefix,
            prefix=state.prefix,
            sort=state.sort,
            same_text=state.same_text,
            provenance=state.provenance,
        )

    def predictions(
        self,
        *,
        offset: int | None = None,
        limit: int | None = None,
        query: str | None = None,
        source_query: str | None = None,
        source_prefix: str | None = None,
        target_query: str | None = None,
        target_prefix: str | None = None,
        prefix: str | None = None,
        sort: str | None = None,
        same_text: bool | None = None,
        provenance: str | None = None,
    ) -> Iterable[tuple[int, Mapping[str, Any]]]:
        """Iterate over predictions.

        :param offset: If given, offset the iteration by this number
        :param limit: If given, only iterate this number of predictions.

        :param query: If given, show only equivalences that have it appearing as a substring in one
            of the source or target fields.

        :param source_query: If given, show only equivalences that have it appearing as a substring
            in one of the source fields.
        :param source_prefix: If given, show only mappings that have it appearing in the source
            prefix field
        :param target_query: If given, show only equivalences that have it appearing as a substring
            in one of the target fields.
        :param target_prefix: If given, show only mappings that have it appearing in the target
            prefix field
        :param prefix: If given, show only equivalences that have it appearing as a substring in one
            of the prefixes.
        :param same_text: If true, filter to predictions with the same label
        :param sort: If "desc", sorts in descending confidence order. If "asc", sorts in increasing
            confidence order. Otherwise, do not sort.
        :param provenance: If given, filters to provenance values matching this
        :yields: Pairs of positions and prediction dictionaries
        """
        if same_text is None:
            same_text = False
        it = self._help_it_predictions(
            query=query,
            source_query=source_query,
            source_prefix=source_prefix,
            target_query=target_query,
            target_prefix=target_prefix,
            prefix=prefix,
            sort=sort,
            same_text=same_text,
            provenance=provenance,
        )
        if offset is not None:
            try:
                for _ in range(offset):
                    next(it)
            except StopIteration:
                # if next() fails, then there are no remaining entries.
                # do not pass go, do not collect 200 euro $
                return
        if limit is None:
            yield from it
        else:
            for line_prediction, _ in zip(it, range(limit), strict=False):
                yield line_prediction

    def count_predictions_from_state(self, state: State) -> int:
        """Count the number of predictions to check for the given filters."""
        return self.count_predictions(
            query=state.query,
            source_query=state.source_query,
            source_prefix=state.source_prefix,
            target_query=state.target_query,
            target_prefix=state.target_prefix,
            prefix=state.prefix,
            same_text=state.same_text,
            provenance=state.provenance,
        )

    def count_predictions(
        self,
        query: str | None = None,
        source_query: str | None = None,
        source_prefix: str | None = None,
        target_query: str | None = None,
        target_prefix: str | None = None,
        prefix: str | None = None,
        sort: str | None = None,
        same_text: bool | None = None,
        provenance: str | None = None,
    ) -> int:
        """Count the number of predictions to check for the given filters."""
        it = self._help_it_predictions(
            query=query,
            source_query=source_query,
            source_prefix=source_prefix,
            target_query=target_query,
            target_prefix=target_prefix,
            prefix=prefix,
            sort=sort,
            same_text=same_text,
            provenance=provenance,
        )
        return sum(1 for _ in it)

    def _help_it_predictions(
        self,
        query: str | None = None,
        source_query: str | None = None,
        source_prefix: str | None = None,
        target_query: str | None = None,
        target_prefix: str | None = None,
        prefix: str | None = None,
        sort: str | None = None,
        same_text: bool | None = None,
        provenance: str | None = None,
    ):
        it: Iterable[tuple[int, Mapping[str, Any]]] = enumerate(self._predictions)
        if self.target_ids:
            it = (
                (line, p)
                for (line, p) in it
                if (p["source prefix"], p["source identifier"]) in self.target_ids
                or (p["target prefix"], p["target identifier"]) in self.target_ids
            )

        if query is not None:
            it = self._help_filter(
                query,
                it,
                {
                    "source prefix",
                    "source identifier",
                    "source name",
                    "target prefix",
                    "target identifier",
                    "target name",
                    "source",
                },
            )
        if source_prefix is not None:
            it = self._help_filter(source_prefix, it, {"source prefix"})
        if source_query is not None:
            it = self._help_filter(
                source_query, it, {"source prefix", "source identifier", "source name"}
            )
        if target_query is not None:
            it = self._help_filter(
                target_query, it, {"target prefix", "target identifier", "target name"}
            )
        if target_prefix is not None:
            it = self._help_filter(target_prefix, it, {"target prefix"})
        if prefix is not None:
            it = self._help_filter(prefix, it, {"source prefix", "target prefix"})
        if provenance is not None:
            it = self._help_filter(provenance, it, {"source"})

        if sort is not None:
            if sort == "desc":
                it = iter(sorted(it, key=lambda l_p: l_p[1]["confidence"], reverse=True))
            elif sort == "asc":
                it = iter(sorted(it, key=lambda l_p: l_p[1]["confidence"], reverse=False))
            elif sort == "object":
                it = iter(
                    sorted(
                        it, key=lambda l_p: (l_p[1]["target prefix"], l_p[1]["target identifier"])
                    )
                )

        if same_text:
            it = (
                (line, prediction)
                for line, prediction in it
                if prediction["source name"].casefold() == prediction["target name"].casefold()
                and prediction["relation"] == "skos:exactMatch"
            )

        return ((line, prediction) for line, prediction in it if line not in self._marked)

    @staticmethod
    def _help_filter(query: str, it, elements: set[str]):
        query = query.casefold()
        return (
            (line, prediction)
            for line, prediction in it
            if any(query in prediction[element].casefold() for element in elements)
        )

    @staticmethod
    def get_curie(prefix: str, identifier: str) -> str:
        """Return CURIE for a given prefix and identifier."""
        return get_curie(prefix, identifier)

    @classmethod
    def get_url(cls, prefix: str, identifier: str) -> str:
        """Return URL for a given prefix and identifier."""
        url = bioregistry.get_bioregistry_iri(prefix, identifier)
        if url is None:
            raise TypeError
        return url

    @property
    def total_predictions(self) -> int:
        """Return the total number of yet unmarked predictions."""
        return len(self._predictions) - len(self._marked)

    def mark(self, line: int, value: str) -> None:
        """Mark the given equivalency as correct.

        :param line: Position of the prediction
        :param value: Value to mark the prediction with
        :raises ValueError: if an invalid value is used
        """
        if line not in self._marked:
            self.total_curated += 1
        if value not in {"correct", "incorrect", "unsure", "broad", "narrow"}:
            raise ValueError
        self._marked[line] = value

    def add_mapping(
        self,
        source_prefix: str,
        source_id: str,
        source_name: str,
        target_prefix: str,
        target_id: str,
        target_name: str,
        user_id: str,
    ) -> None:
        """Add manually curated new mappings."""
        try:
            check_valid_prefix_id(source_prefix, source_id)
        except ValueError as e:
            flask.flash(
                f"Problem with source CURIE {source_prefix}:{source_id}: {e.__class__.__name__}",
                category="warning",
            )
            return

        try:
            check_valid_prefix_id(target_prefix, target_id)
        except ValueError as e:
            flask.flash(
                f"Problem with target CURIE {target_prefix}:{target_id}: {e.__class__.__name__}",
                category="warning",
            )
            return

        self._added_mappings.append(
            {
                "source prefix": source_prefix,
                "source identifier": source_id,
                "source name": source_name,
                "relation": "skos:exactMatch",
                "target prefix": target_prefix,
                "target identifier": target_id,
                "target name": target_name,
                "source": user_id,
                "type": "manual",
                "prediction_type": None,
                "prediction_source": None,
                "prediction_confidence": None,
            }
        )
        self.total_curated += 1

    def persist(self):
        """Save the current markings to the source files."""
        state = State.from_flask_globals()
        entries = defaultdict(list)

        for line, value in sorted(self._marked.items(), reverse=True):
            prediction = self._predictions.pop(line)
            prediction["prediction_type"] = prediction.pop("type")
            prediction["prediction_source"] = prediction.pop("source")
            prediction["prediction_confidence"] = prediction.pop("confidence")
            prediction["source"] = state.user_id
            prediction["type"] = "semapv:ManualMappingCuration"

            # note these go backwards because of the way they are read
            if value == "broad":
                value = "correct"  # noqa: PLW2901
                prediction["relation"] = "skos:narrowMatch"
            elif value == "narrow":
                value = "correct"  # noqa: PLW2901
                prediction["relation"] = "skos:broadMatch"

            entries[value].append(prediction)

        append_true_mappings(entries["correct"], path=self.positives_path)
        append_false_mappings(entries["incorrect"], path=self.negatives_path)
        append_unsure_mappings(entries["unsure"], path=self.unsure_path)
        write_predictions(self._predictions, path=self.predictions_path)
        self._marked.clear()

        # Now add manually curated mappings
        append_true_mappings(self._added_mappings, path=self.positives_path)
        self._added_mappings = []


CONTROLLER: Controller = LocalProxy(lambda: current_app.config["controller"])  # type: ignore[assignment]


class MappingForm(FlaskForm):
    """Form for entering new mappings."""

    source_prefix = StringField("Source Prefix", id="source_prefix")
    source_id = StringField("Source ID", id="source_id")
    source_name = StringField("Source Name", id="source_name")
    target_prefix = StringField("Target Prefix", id="target_prefix")
    target_id = StringField("Target ID", id="target_id")
    target_name = StringField("Target Name", id="target_name")
    submit = SubmitField("Add")


blueprint = flask.Blueprint("ui", __name__)


@blueprint.route("/")
def home():
    """Serve the home page."""
    state = State.from_flask_globals()
    form = MappingForm()
    predictions = CONTROLLER.predictions_from_state(state)
    remaining_rows = CONTROLLER.count_predictions_from_state(state)
    return flask.render_template(
        "home.html",
        predictions=predictions,
        form=form,
        state=state,
        remaining_rows=remaining_rows,
    )


@blueprint.route("/summary")
def summary():
    """Serve the summary page."""
    state = State.from_flask_globals()
    state.limit = None
    predictions = CONTROLLER.predictions_from_state(state)
    counter = Counter(
        (mapping["source prefix"], mapping["target prefix"]) for _, mapping in predictions
    )
    rows = []
    for (source_prefix, target_prefix), count in counter.most_common():
        row_state = deepcopy(state)
        row_state.source_prefix = source_prefix
        row_state.target_prefix = target_prefix
        rows.append((source_prefix, target_prefix, count, url_for_state(".home", row_state)))

    return flask.render_template(
        "summary.html",
        state=state,
        rows=rows,
    )


@blueprint.route("/add_mapping", methods=["POST"])
def add_mapping():
    """Add a new mapping manually."""
    form = MappingForm()
    if form.is_submitted():
        state = State.from_flask_globals()
        CONTROLLER.add_mapping(
            form.data["source_prefix"],
            form.data["source_id"],
            form.data["source_name"],
            form.data["target_prefix"],
            form.data["target_id"],
            form.data["target_name"],
            state.user_id,
        )
        CONTROLLER.persist()
    else:
        flask.flash("missing form data", category="warning")
    return _go_home()


@blueprint.route("/commit")
def run_commit():
    """Make a commit then redirect to the home page."""
    state = State.from_flask_globals()
    commit_info = commit(
        f"Curated {CONTROLLER.total_curated} mapping"
        f"{'s' if CONTROLLER.total_curated > 1 else ''}"
        f" ({state.user_id})",
    )
    current_app.logger.warning("git commit res: %s", commit_info)
    if not_main():
        branch = get_branch()
        push_output = push(branch_name=branch)
        current_app.logger.warning("git push res: %s", push_output)
    else:
        flask.flash("did not push because on master branch")
        current_app.logger.warning("did not push because on master branch")
    CONTROLLER.total_curated = 0
    return _go_home()


CORRECT = {"yup", "true", "t", "correct", "right", "close enough", "disco"}
INCORRECT = {"no", "nope", "false", "f", "nada", "nein", "incorrect", "negative", "negatory"}
UNSURE = {"unsure", "maybe", "idk", "idgaf", "idgaff"}


def _normalize_mark(value: str) -> str:
    value = value.lower()
    if value in CORRECT:
        return "correct"
    if value in INCORRECT:
        return "incorrect"
    if value in UNSURE:
        return "unsure"
    if value in {"broader", "broad"}:
        return "broad"
    if value in {"narrow", "narrower"}:
        return "narrow"
    raise ValueError


@blueprint.route("/mark/<int:line>/<value>")
def mark(line: int, value: str):
    """Mark the given line as correct or not."""
    CONTROLLER.mark(line, _normalize_mark(value))
    CONTROLLER.persist()
    return _go_home()


def _go_home():
    state = State.from_flask_globals()
    return flask.redirect(url_for_state(".home", state))


app = get_app(
    predictions_path=RESOURCES_DIR.joinpath("predictions.tsv"),
    positives_path=RESOURCES_DIR.joinpath("mappings.tsv"),
    negatives_path=RESOURCES_DIR.joinpath("incorrect.tsv"),
    unsure_path=RESOURCES_DIR.joinpath("unsure.tsv"),
)

if __name__ == "__main__":
    app.run()
