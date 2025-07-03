"""Web curation interface for :mod:`biomappings`."""

from __future__ import annotations

import datetime
import functools
import itertools
import operator
import os
import shutil
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast, get_args
from urllib.parse import quote_plus

import flask
import flask_bootstrap
import stamina
import werkzeug
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from httpx import (
    URL,
    Auth,
    Client,
    Headers,
    HTTPError,
    HTTPStatusError,
    Request,
    Response,
    Timeout,
    codes,
)
from markupsafe import Markup
from pydantic import BaseModel, ValidationError
from sqlalchemy.dialects.postgresql import insert as postgres_upsert
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column
from sqlalchemy.schema import MetaData, PrimaryKeyConstraint
from werkzeug.local import LocalProxy
from werkzeug.middleware.proxy_fix import ProxyFix
from wtforms import StringField, SubmitField

from biomappings.resources import (
    SemanticMapping,
    append_false_mappings,
    append_true_mappings,
    append_unsure_mappings,
    load_predictions,
    write_predictions,
)
from biomappings.utils import (
    BROAD_MATCH,
    EXACT_MATCH,
    MANUAL_MAPPING_CURATION,
    NARROW_MATCH,
)
from bioregistry import NormalizedNamableReference, get_resource
from curies import NamableReference

MarkType = Literal["correct", "incorrect", "unsure", "broad", "narrow"]

AUTHOR_EMAIL = os.environ["COMMITTER_EMAIL"]
BASE_BRANCH = os.environ["BASE_BRANCH"]
COMMITTER_EMAIL = os.environ["COMMITTER_EMAIL"]
COMMITTER_NAME = os.environ["COMMITTER_NAME"]
GITHUB_API_BASE_URL = URL(os.environ["GITHUB_API_BASE_URL"])
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
LOGIN_REQUIRED_MSG = "Login required"
MARKS: set[MarkType] = set(get_args(MarkType))
NUM_PROXIES = int(os.environ["NUM_PROXIES"])
NUM_RETRIES = 3
SQLALCHEMY_DATABASE_URI = URL(os.environ["SQLALCHEMY_DATABASE_URI"])
TIMEOUT = datetime.timedelta(seconds=3)


class BearerTokenAuth(Auth):
    def __init__(self, token: str) -> None:
        self.token = token

    def auth_flow(self, request: Request) -> Generator[Request, Response]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


def is_request_or_server_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.response.is_server_error
    return isinstance(exc, HTTPError)


BiomappingsApiClient = functools.partial(
    Client,
    auth=BearerTokenAuth(GITHUB_TOKEN),
    base_url=GITHUB_API_BASE_URL,
    headers=Headers(
        {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    ),
    http2=True,
    timeout=Timeout(TIMEOUT.total_seconds()),
)


@stamina.retry(on=is_request_or_server_error, attempts=NUM_RETRIES)
def create_pull_request(*, client: Client, base: str, head: str, title: str, body: str) -> URL:
    response = client.post(
        "/pulls",
        json={
            "base": base,
            "body": body,
            "head": head,
            "maintainer_can_modify": True,
            "title": title,
        },
    )
    response.raise_for_status()
    return URL(response.json()["html_url"])


@stamina.retry(on=is_request_or_server_error, attempts=NUM_RETRIES)
def delete_branch_if_exists(*, client: Client, head: str) -> None:
    response = client.delete(f"/git/refs/heads/{head}")
    if response.is_success or (
        response.status_code == codes.UNPROCESSABLE_ENTITY
        and response.json()["message"] == "Reference does not exist"
    ):
        return
    response.raise_for_status()


def startswith(string: str, prefix: str) -> bool:
    return string.startswith(prefix)


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


class Mark(db.Model):  # type: ignore[name-defined]
    __tablename__ = "mark"

    user_id: Mapped[str]
    line: Mapped[int]
    value: Mapped[str]

    __table_args__ = (PrimaryKeyConstraint("user_id", "line"),)


class UserMeta(db.Model):  # type: ignore[name-defined]
    __tablename__ = "user_meta"

    user_id: Mapped[str]
    total_curated: Mapped[int] = mapped_column(default=0)

    __table_args__ = (PrimaryKeyConstraint("user_id"),)


class Mapping(db.Model):  # type: ignore[name-defined]
    __tablename__ = "mapping"

    kind: Mapped[str]
    line: Mapped[int | None]
    subject_id: Mapped[str]
    subject_label: Mapped[str]
    predicate_id: Mapped[str]
    object_id: Mapped[str]
    object_label: Mapped[str]
    author_id: Mapped[str]
    mapping_justification: Mapped[str]
    mapping_tool: Mapped[str | None]
    confidence: Mapped[float | None]
    predicate_modifier: Mapped[str | None]

    __table_args__ = (
        PrimaryKeyConstraint(
            "author_id", "subject_id", "subject_label", "object_id", "object_label"
        ),
    )


class PublishedMark(db.Model):  # type: ignore[name-defined]
    __tablename__ = "published_mark"

    user_id: Mapped[str]
    line: Mapped[int]
    value: Mapped[str]

    __table_args__ = (PrimaryKeyConstraint("user_id", "line"),)


class State(BaseModel):
    """Contains the state for queries to the curation app."""

    limit: int | None = 20
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

    @classmethod
    def from_flask_globals(cls) -> State:
        """Get the state from the flask current request."""
        return State(
            limit=flask.request.args.get("limit", type=int, default=20),
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
        )


def _get_bool_arg(name: str, default: bool | None = None) -> bool | None:  # noqa: FBT001
    value = flask.request.args.get(name, type=str)
    if value is not None:
        return value.lower() in {"true", "t"}
    return default


def url_for_state(endpoint, state: State, **kwargs: Any) -> str:
    """Get the URL for an endpoint based on the state class."""
    vv = state.model_dump(exclude_none=True, exclude_defaults=True)
    vv.update(kwargs)  # make sure stuff explicitly set overrides state
    return flask.url_for(endpoint, **vv)


def get_app(biomappings_path: Path) -> flask.Flask:
    """Get a curation flask app."""
    app_ = flask.Flask(__name__)
    app_.config["SECRET_KEY"] = os.urandom(8)
    app_.config["SHOW_LINES"] = False
    app_.config["SHOW_RELATIONS"] = True
    app_.config["SQLALCHEMY_DATABASE_URI"] = str(SQLALCHEMY_DATABASE_URI)
    app_.config["WTF_CSRF_ENABLED"] = False
    controller = Controller(biomappings_path=biomappings_path)
    app_.config["controller"] = controller
    flask_bootstrap.Bootstrap4(app_)
    app_.register_blueprint(blueprint)
    app_.jinja_env.filters["quote_plus"] = quote_plus
    app_.jinja_env.globals.update(controller=controller, url_for_state=url_for_state)
    app_.wsgi_app = ProxyFix(  # type: ignore[method-assign]
        app_.wsgi_app,
        x_for=NUM_PROXIES,
        x_proto=NUM_PROXIES,
        x_host=NUM_PROXIES,
    )
    db.init_app(app_)
    if int(os.environ["APP_WORKER_ID"]) == 0:
        with app_.app_context():
            db.create_all()
    return app_


class Controller:
    """A module for interacting with the predictions and mappings."""

    def __init__(self, *, biomappings_path: Path) -> None:
        """Instantiate the web controller.

        :param biomappings_path: path to the Biomappings Git repository
        """
        self.biomappings_path = biomappings_path
        self._predictions = load_predictions(
            path=biomappings_path.joinpath(
                "src", "biomappings", "resources", "predictions.sssom.tsv"
            ),
        )

    def predictions_from_state(self, state: State) -> Iterable[tuple[int, SemanticMapping]]:
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
            user_id=self.user_id,
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
        user_id: str | None = None,
    ) -> Iterable[tuple[int, SemanticMapping]]:
        """Iterate over predictions.

        :param offset: If given, offset the iteration by this number
        :param limit: If given, only iterate this number of predictions.

        :param query: If given, show only equivalences that have it appearing as a substring in one
            of the source or target fields.

        :param source_query: If given, show only equivalences that have it appearing as a substring
            in one of the source fields.
        :param source_prefix: If given, show only mappings that have it equaling the source prefix
            field
        :param target_query: If given, show only equivalences that have it appearing as a substring
            in one of the target fields.
        :param target_prefix: If given, show only mappings that have it equaling the target prefix
            field
        :param prefix: If given, show only equivalences that have it equaling one of the prefixes.
        :param same_text: If true, filter to predictions with the same label
        :param sort: If "desc", sorts in descending confidence order. If "asc", sorts in increasing
            confidence order. Otherwise, do not sort.
        :param provenance: If given, filters to provenance values matching this
        :param user_id: If given, exclude predictions marked by this authenticated user ID.
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
            user_id=user_id,
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
            user_id=self.user_id,
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
        same_text: bool | None = None,  # noqa: FBT001
        provenance: str | None = None,
        user_id: str | None = None,
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
            user_id=user_id,
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
        same_text: bool | None = None,  # noqa: FBT001
        provenance: str | None = None,
        user_id: str | None = None,
    ) -> Iterator[tuple[int, SemanticMapping]]:
        it: Iterable[tuple[int, SemanticMapping]] = enumerate(self._predictions)

        if query is not None:
            it = self._help_filter(
                query,
                it,
                lambda mapping: [
                    mapping.subject.curie,
                    mapping.subject.name,
                    mapping.object.curie,
                    mapping.object.name,
                    mapping.mapping_tool,
                ],
            )
        if source_prefix is not None:
            it = self._help_filter(
                f"{source_prefix}:",
                it,
                lambda mapping: [mapping.subject.curie],
                op_element_query=startswith,
            )
        if source_query is not None:
            it = self._help_filter(
                source_query,
                it,
                lambda mapping: [mapping.subject.curie, mapping.subject.name],
            )
        if target_query is not None:
            it = self._help_filter(
                target_query,
                it,
                lambda mapping: [mapping.object.curie, mapping.object.name],
            )
        if target_prefix is not None:
            it = self._help_filter(
                f"{target_prefix}:",
                it,
                lambda mapping: [mapping.object.curie],
                op_element_query=startswith,
            )
        if prefix is not None:
            it = self._help_filter(
                f"{prefix}:",
                it,
                lambda mapping: [mapping.subject.curie, mapping.object.curie],
                op_element_query=startswith,
            )
        if provenance is not None:
            it = self._help_filter(provenance, it, lambda mapping: [mapping.mapping_tool])

        def _get_confidence(t: tuple[int, SemanticMapping]) -> float:
            return t[1].confidence or 0.0

        if sort is not None:
            if sort == "desc":
                it = iter(sorted(it, key=_get_confidence, reverse=True))
            elif sort == "asc":
                it = iter(sorted(it, key=_get_confidence, reverse=False))
            elif sort == "object":
                it = iter(sorted(it, key=lambda l_p: l_p[1].object.curie))
            else:
                msg = f"unknown sort type: {sort}"
                raise ValueError(msg)

        if same_text:
            it = (
                (line, mapping)
                for line, mapping in it
                if mapping.subject.name is not None
                and mapping.object.name is not None
                and mapping.subject.name.casefold() == mapping.object.name.casefold()
                and mapping.predicate.curie == "skos:exactMatch"
            )

        marked = set()
        if user_id is not None:
            marked = set(
                map(
                    operator.itemgetter(0),
                    db.session.query(Mark.line).filter(Mark.user_id == user_id),
                )
            )

        return ((line, mapping) for line, mapping in it if line not in marked)

    @staticmethod
    def _help_filter(
        query: str,
        it: Iterable[tuple[int, SemanticMapping]],
        func: Callable[[SemanticMapping], list[str | None]],
        op_element_query: Callable[[str, str], bool] = operator.contains,
    ) -> Iterable[tuple[int, SemanticMapping]]:
        query = query.casefold()
        for line, mapping in it:
            if any(
                op_element_query(element.casefold(), query)
                for element in func(mapping)
                if element is not None
            ):
                yield line, mapping

    @classmethod
    def get_prefix_display_name(cls, prefix: str) -> str:
        """Return display name for a given prefix."""
        resource = get_resource(prefix)
        if resource is None:
            raise TypeError
        if (name := resource.get_name()) is not None:
            return name
        return prefix

    @classmethod
    def get_logo_url(cls, prefix: str) -> str | None:
        """Return logo URL for a given prefix."""
        resource = get_resource(prefix)
        if resource is None:
            raise TypeError
        return resource.get_logo()

    @property
    def total_predictions(self) -> int:
        """Return the total number of yet unmarked predictions."""
        mark_count = 0
        if (user_id := self.user_id) is not None:
            mark_count = db.session.query(Mark).filter(Mark.user_id == user_id).count()
        return len(self._predictions) - mark_count

    def mark(self, user_id: str, line: int, value: MarkType) -> None:
        """Mark the given equivalency as correct.

        :param user_id: Authenticated user ID
        :param line: Position of the prediction
        :param value: Value to mark the prediction with
        :raises ValueError: if an invalid value is used
        """
        if line > len(self._predictions):
            msg = (
                f"given line {line} is larger than the number of predictions "
                f"{len(self._predictions):,}"
            )
            raise IndexError(msg)
        mark_ = db.session.get(Mark, (user_id, line))
        if mark_ is None:
            user_meta = db.session.get(UserMeta, user_id) or UserMeta(user_id=user_id)
            user_meta.total_curated += 1
            db.session.add(user_meta)
        if value not in MARKS:
            msg = f"illegal mark value given: {value}. Should be one of {MARKS}"
            raise ValueError(msg)
        db.session.add(Mark(user_id=user_id, line=line, value=value))
        db.session.commit()

    @staticmethod
    def add_mapping(
        subject: NormalizedNamableReference,
        obj: NormalizedNamableReference,
        user_id: str,
    ) -> None:
        """Add manually curated new mappings."""
        db.session.add(
            Mapping(
                kind="correct",
                line=None,
                subject_id=subject.curie,
                subject_label=subject.name,
                predicate_id=EXACT_MATCH.curie,
                object_id=obj.curie,
                object_label=obj.name,
                author_id=user_id,
                mapping_justification=MANUAL_MAPPING_CURATION.curie,
                mapping_tool=None,
                confidence=None,
                predicate_modifier=None,
            )
        )

        user_meta = db.session.get(UserMeta, user_id) or UserMeta(user_id=user_id)
        user_meta.total_curated += 1
        db.session.add(user_meta)

        db.session.commit()

    def persist(self, user_id):
        """Save the current markings to the source files."""
        marks = dict(
            db.session.query(Mark.line, Mark.value)
            .filter(Mark.user_id == user_id)
            .outerjoin(Mapping, (Mark.user_id == Mapping.author_id) & (Mark.line == Mapping.line))
            .filter(Mapping.line.is_(None))
        )
        mappings = []

        for line, value in sorted(marks.items(), reverse=True):
            try:
                mapping = deepcopy(self._predictions[line])
            except IndexError as exc:
                msg = (
                    f"you tried popping the {line} element from the predictions list, which only "
                    f"has {len(self._predictions):,} elements"
                )
                raise IndexError(msg) from exc

            predicate = mapping.predicate
            predicate_modifier = None
            # note these go backwards because of the way they are read
            if value == "broad":
                kind = "correct"
                predicate = NARROW_MATCH
            elif value == "narrow":
                kind = "correct"
                predicate = BROAD_MATCH
            elif value == "incorrect":
                kind = "incorrect"
                predicate_modifier = "Not"
            elif value == "correct":
                kind = "correct"
            elif value == "unsure":
                kind = "unsure"
            else:
                raise NotImplementedError

            mappings.append(
                Mapping(
                    kind=kind,
                    line=line,
                    subject_id=mapping.subject.curie,
                    subject_label=mapping.subject.name,
                    predicate_id=predicate.curie,
                    object_id=mapping.object.curie,
                    object_label=mapping.object.name,
                    author_id=user_id,
                    mapping_justification=MANUAL_MAPPING_CURATION.curie,
                    mapping_tool=mapping.mapping_tool,
                    confidence=mapping.confidence,
                    predicate_modifier=predicate_modifier,
                )
            )

        db.session.add_all(mappings)
        db.session.commit()

    def clear_user_state(self, user_id: str) -> None:
        """Clear user-controlled state."""
        self._clear_user_state_no_commit(user_id)
        db.session.commit()

    def update_user_state_after_publish(self, user_id: str) -> None:
        """Update user-specific state after publishing PR."""
        if marks := db.session.query(Mark).filter(Mark.user_id == user_id).all():
            stmt = postgres_upsert(PublishedMark).values(
                [
                    {"user_id": mark.user_id, "line": mark.line, "value": mark.value}
                    for mark in marks
                ]
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[PublishedMark.user_id, PublishedMark.line],
                set_={"value": stmt.excluded.value},
            )
            db.session.execute(stmt)
        self._clear_user_state_no_commit(user_id)
        db.session.commit()

    @staticmethod
    def _clear_user_state_no_commit(user_id: str) -> None:
        """Clear user-controlled state, but do not commit."""
        db.session.query(Mark).filter(Mark.user_id == user_id).delete()
        db.session.query(UserMeta).filter(UserMeta.user_id == user_id).delete()
        db.session.query(Mapping).filter(Mapping.author_id == user_id).delete()

    def get_all_mappings(self, user_id: str):
        true_mappings = []
        false_mappings = []
        unsure_mappings = []
        marked = set()

        for mapping in db.session.query(Mapping).filter(Mapping.author_id == user_id):
            kind = mapping.kind
            line = mapping.line
            mapping_ = SemanticMapping(
                subject=NamableReference.from_curie(mapping.subject_id, name=mapping.subject_label),
                predicate=NamableReference.from_curie(mapping.predicate_id),
                object=NamableReference.from_curie(mapping.object_id, name=mapping.object_label),
                mapping_justification=NamableReference.from_curie(mapping.mapping_justification),
                author=NamableReference.from_curie(mapping.author_id),
                mapping_tool=mapping.mapping_tool,
                predicate_modifier=mapping.predicate_modifier,  # type: ignore[arg-type]
                confidence=mapping.confidence,
            )
            if kind == "correct":
                true_mappings.append(mapping_)
            elif kind == "incorrect":
                false_mappings.append(mapping_)
            elif kind == "unsure":
                unsure_mappings.append(mapping_)
            else:
                raise ValueError
            if line is not None:
                marked.add(line)

        return (
            true_mappings,
            false_mappings,
            unsure_mappings,
            (mapping_ for line, mapping_ in enumerate(self._predictions) if line not in marked),
        )

    @property
    def user_id(self) -> str | None:
        if (value := flask.request.headers.get("X-Auth-Request-User")) is None:
            return None
        return f"orcid:{value}"

    @property
    def logged_in(self) -> bool:
        return self.user_id is not None

    def is_published(self, line: int) -> bool:
        if (user_id := self.user_id) is None:
            return False
        published_mark = db.session.get(PublishedMark, (user_id, line))
        return published_mark is not None


CONTROLLER: Controller = cast(Controller, LocalProxy(lambda: current_app.config["controller"]))


class MappingForm(FlaskForm):
    """Form for entering new mappings."""

    subject_prefix = StringField("Subject Prefix", id="subject_prefix")
    subject_id = StringField("Subject ID", id="subject_id")
    subject_name = StringField("Subject Label", id="subject_name")
    object_prefix = StringField("Object Prefix", id="object_prefix")
    object_id = StringField("Object ID", id="object_id")
    object_name = StringField("Object Label", id="object_name")
    submit = SubmitField("Add")

    def get_subject(self) -> NormalizedNamableReference:
        """Get the subject."""
        return NormalizedNamableReference(
            prefix=self.data["subject_prefix"],
            identifier=self.data["subject_id"],
            name=self.data["subject_name"],
        )

    def get_object(self) -> NormalizedNamableReference:
        """Get the object."""
        return NormalizedNamableReference(
            prefix=self.data["object_prefix"],
            identifier=self.data["object_id"],
            name=self.data["object_name"],
        )


blueprint = flask.Blueprint("ui", __name__)


@blueprint.route("/home")
def home() -> str:
    """Serve the home page."""
    state = State.from_flask_globals()
    form = MappingForm()
    predictions = CONTROLLER.predictions_from_state(state)
    remaining_rows = CONTROLLER.count_predictions_from_state(state)
    total_curated = 0
    if (user_id := CONTROLLER.user_id) is not None:
        total_curated = (
            db.session.get(UserMeta, user_id) or UserMeta(user_id=user_id)
        ).total_curated
    return flask.render_template(
        "home.html",
        predictions=predictions,
        form=form,
        state=state,
        remaining_rows=remaining_rows,
        total_curated=total_curated,
    )


@blueprint.route("/")
def summary() -> str:
    """Serve the summary page."""
    state = State.from_flask_globals()
    state.limit = None
    predictions = CONTROLLER.predictions_from_state(state)
    counter = Counter(
        itertools.chain.from_iterable(
            (mapping.subject.prefix, mapping.object.prefix) for _, mapping in predictions
        )
    )
    rows = []
    for prefix, count in counter.most_common():
        row_state = deepcopy(state)
        row_state.prefix = prefix
        display_name = CONTROLLER.get_prefix_display_name(prefix)
        logo_url = CONTROLLER.get_logo_url(prefix)
        rows.append((prefix, count, url_for_state(".home", row_state), display_name, logo_url))

    return flask.render_template(
        "summary.html",
        state=state,
        rows=rows,
    )


@blueprint.route("/add_mapping", methods=["POST"])
def add_mapping() -> werkzeug.Response:
    """Add a new mapping manually."""
    if (user_id := CONTROLLER.user_id) is None:
        flask.flash(LOGIN_REQUIRED_MSG, category="warning")
    else:
        form = MappingForm()
        if form.is_submitted():
            try:
                subject = form.get_subject()
            except ValidationError as e:
                flask.flash(f"Problem with subject CURIE {e}", category="warning")
                return _go_home()

            try:
                obj = form.get_object()
            except ValidationError as e:
                flask.flash(f"Problem with object CURIE {e}", category="warning")
                return _go_home()

            CONTROLLER.add_mapping(subject, obj, user_id)
        else:
            flask.flash("missing form data", category="warning")
    return _go_home()


@blueprint.route("/add_mapping")
def _add_mapping() -> werkzeug.Response:
    """Handle when POST method becomes a GET after auth redirections."""
    flask.flash(
        (
            "It's likely your login credentials had expired before submitting your custom mapping. "
            "Please try adding the mapping again."
        ),
        category="warning",
    )
    return _go_home()


@blueprint.route("/clear_user_state")
def clear_user_state() -> werkzeug.Response:
    """Clear all user-specific state, then redirect to the home page."""
    if (user_id := CONTROLLER.user_id) is None:
        flask.flash(LOGIN_REQUIRED_MSG, category="warning")
    else:
        CONTROLLER.clear_user_state(user_id)
    return _go_home()


@blueprint.route("/publish")
def publish_pr() -> werkzeug.Response:
    """Publish a PR, then clear user state and redirect to the home page."""
    if (user_id := CONTROLLER.user_id) is None:
        flask.flash(LOGIN_REQUIRED_MSG, category="warning")
        return _go_home()

    true_mappings, false_mappings, unsure_mappings, predicted_mappings = (
        CONTROLLER.get_all_mappings(user_id)
    )
    total_curated = len(true_mappings) + len(false_mappings) + len(unsure_mappings)

    head = f"{user_id}_{uuid.uuid4()}".replace(":", "_")
    author = f"{user_id} <{AUTHOR_EMAIL}>"
    commit_msg = (
        f"Curated {total_curated} mapping{'s' if total_curated > 1 else ''} via Biomappings web app"
    )
    title = commit_msg
    body = (
        f"These mappings were curated via the Biomappings web app by "
        f"[{user_id}](https://bioregistry.io/{quote_plus(user_id)})."
    )

    with TemporaryDirectory() as _tmp_path:
        tmp_path = Path(_tmp_path)
        shutil.copytree(
            CONTROLLER.biomappings_path,
            tmp_path,
            dirs_exist_ok=True,
            ignore_dangling_symlinks=True,
        )
        shutil.rmtree(tmp_path.joinpath(".git", "hooks"), ignore_errors=True)

        resources_dir = tmp_path.joinpath("src", "biomappings", "resources")
        true_path = resources_dir.joinpath("positive.sssom.tsv")
        false_path = resources_dir.joinpath("negative.sssom.tsv")
        unsure_path = resources_dir.joinpath("unsure.sssom.tsv")
        predictions_path = resources_dir.joinpath("predictions.sssom.tsv")

        append_true_mappings(true_mappings, path=true_path, sort=True, standardize=False)
        append_false_mappings(false_mappings, path=false_path, sort=True, standardize=False)
        append_unsure_mappings(unsure_mappings, path=unsure_path, sort=True, standardize=False)
        write_predictions(predicted_mappings, path=predictions_path)

        run = functools.partial(
            subprocess.run,
            check=True,
            cwd=tmp_path,
            timeout=TIMEOUT.total_seconds(),
        )

        run(["git", "switch", "-c", head])
        run(["git", "config", "set", "--local", "--", "user.name", COMMITTER_NAME])
        run(["git", "config", "set", "--local", "--", "user.email", COMMITTER_EMAIL])
        run(["git", "commit", "--all", "--author", author, "-m", commit_msg])

        with BiomappingsApiClient() as client:
            try:
                run(["git", "push", "--", "origin", head])
                pull_request_url = create_pull_request(
                    client=client, base=BASE_BRANCH, head=head, title=title, body=body
                )
            except Exception:
                delete_branch_if_exists(client=client, head=head)
                raise

    CONTROLLER.update_user_state_after_publish(user_id)
    flask.flash(Markup('PR submitted <a href="{href}">here</a>!').format(href=pull_request_url))
    return _go_home()


CORRECT = {"yup", "true", "t", "correct", "right", "close enough", "disco"}
INCORRECT = {"no", "nope", "false", "f", "nada", "nein", "incorrect", "negative", "negatory"}
UNSURE = {"unsure", "maybe", "idk", "idgaf", "idgaff"}


def _normalize_mark(value: str) -> MarkType:
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
def mark(line: int, value: str) -> werkzeug.Response:
    """Mark the given line as correct or not."""
    if (user_id := CONTROLLER.user_id) is None:
        flask.flash(LOGIN_REQUIRED_MSG, category="warning")
    else:
        CONTROLLER.mark(user_id, line, _normalize_mark(value))
        CONTROLLER.persist(user_id)
    return _go_home()


def _go_home() -> werkzeug.Response:
    state = State.from_flask_globals()
    return flask.redirect(url_for_state(".home", state))


app = get_app(biomappings_path=Path("biomappings"))


if __name__ == "__main__":
    app.run()
