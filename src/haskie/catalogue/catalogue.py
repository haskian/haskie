"""The model catalogue: every model the runtimes can load, its metadata, and the embedding
profiles a user picks from. It lives in the database (`models`, `embedding_profiles`), seeded once
from `seed.sql`, so this module reads it and holds none of it.

How a model loads stays in code: the pinned revisions in `onnx_models`, `mlx_models` and
`gguf_models` name reviewed code and weights, and a row cannot add a loader. Settings name a
profile and a reranker model by key, and those keys are checked here, because a settings struct
decodes without the database: at the write boundaries (`check`), and when the stored row is read
(`unknown`).
"""

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import msgspec
from sqlalchemy import Row, Select, func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from haskie import db, home
from haskie.errors import InvalidInput
from haskie.indexing import gguf_models, hardware
from haskie.indexing.hardware import Device, Runtime
from haskie.settings import (
    NO_EMBEDDING,
    Accelerator,
    CollectionOverrides,
    Reranker,
    UserSettings,
)
from haskie.tables import embedding_profiles, models, reranker_calibration


class ModelMetadata(msgspec.Struct, frozen=True):
    """What the catalogue says about one model, so it can be chosen without looking it up. The
    runtime and devices come from the loaders (`indexing.hardware`), the rest from the database."""

    description: str
    parameters: int  # the published weights' own total: what the pickers sort by
    context_tokens: int  # the longest input the model states it reads; a loader may cut shorter
    languages: str
    license: str
    released: date  # the original weights' first commit on the Hugging Face Hub
    model_card_url: str  # the original model's card, also for a conversion of it
    runtime: Runtime
    devices: tuple[Device, ...]  # every device it can run on, whether or not this machine has it


class EmbedderMetadata(ModelMetadata, frozen=True):
    dimensions: int  # of the vectors stored and searched: the model's own, or its Matryoshka cut


class RerankerMetadata(ModelMetadata, frozen=True):
    """A reranker's metadata: the facts every model has, and none of its own."""


class RerankerCalibration(msgspec.Struct, frozen=True):
    """How one reranker's scores read (`reranker_calibration`): measured on borderline pairs by
    `catalogue.calibrate`, else the seed's uncalibrated defaults."""

    floor: float  # under it the reranker judged a chunk no answer (`min_rerank_score`)
    beta_a: float  # the beta curve that spreads its scores evenly over 0 to 1 (`fill.absolute`)
    beta_b: float
    source: str  # what measured it, or "uncalibrated"


UNCALIBRATED = RerankerCalibration(floor=0.05, beta_a=1.0, beta_b=1.0, source="uncalibrated")


class DuplicateCosine(msgspec.Struct, frozen=True):
    """The raw cosines two search results must exceed to count as the same point
    (`search.collapse`). Exceeded, not reached: Set-Encoder's near-duplicate is Jaccard > 0.5."""

    chunk: float  # chunk to chunk, and containment
    passage: float  # mean vector to mean vector: means are smoother, so they run higher


class EmbeddingModel(msgspec.Struct):
    name: str
    dims: int  # of the vectors stored and searched: the model's own, or its Matryoshka cut
    accelerator: Accelerator = Accelerator.AUTO
    duplicate: DuplicateCosine | None = None  # None: search results are compared by words alone
    # The catalogue key it was chosen by: one model cut two ways is two profiles, and a query
    # vector compares only with another of the same profile (`search.gaps`).
    profile: str = ""
    # The cosines the Gaps page judges by (`search.gaps`); None turns that judgement off. A best
    # match under `weak_match` is no answer; two queries over `same_topic` ask about one thing.
    weak_match: float | None = None
    # at and over it a best match is an answer; between the two bars it is borderline
    answered_match: float | None = None
    same_topic: float | None = None
    # What the model was trained to read ahead of a query and of a passage (e5's "query: " and
    # "passage: "); empty for models that need none. They shape every vector, so
    # changing one is changing the model. The document prefix is part of `cache_name`, so the
    # cached vectors it shaped are not served after it changes.
    query_prefix: str = ""
    document_prefix: str = ""
    # A model trained so that the first values of its vector are a vector of their own
    # (Matryoshka Representation Learning): its vectors are cut to `dims`, then normalized again,
    # which makes the index and every comparison smaller for a small loss.
    matryoshka: bool = False

    @property
    def cache_name(self) -> str:
        """What the embedding cache keys this model's vectors by: everything that shapes a stored
        vector. Its name and size are readable; the document prefix and the Matryoshka cut are
        hashed. The query prefix, the accelerator and the thresholds shape no stored vector,
        so changing them keeps the cache."""
        shaping = msgspec.json.encode([self.document_prefix, self.matryoshka])
        return f"{self.name}@{self.dims}:{hashlib.sha256(shaping).hexdigest()[:12]}"


# the columns of `models` a `ModelMetadata` reads but the description, which a profile may override
_METADATA = (
    models.c.name,
    models.c.parameters,
    models.c.context_tokens,
    models.c.languages,
    models.c.license,
    models.c.released,
    models.c.model_card_url,
)
_PROFILE = (
    embedding_profiles.c.profile,
    embedding_profiles.c.model,
    embedding_profiles.c.dims,
    embedding_profiles.c.query_prefix,
    embedding_profiles.c.document_prefix,
    embedding_profiles.c.matryoshka,
    embedding_profiles.c.duplicate_chunk,
    embedding_profiles.c.duplicate_passage,
    embedding_profiles.c.weak_match,
    embedding_profiles.c.answered_match,
    embedding_profiles.c.same_topic,
)


def _profiles(*columns: Any) -> Select[Any]:
    """`columns` of every profile joined to its model, in the order every picker lists them: the
    shorter vectors first, and at one size the smaller model first. The profile key breaks a tie,
    so the order never depends on the insert order."""
    return (
        select(*columns)
        .join_from(embedding_profiles, models, models.c.name == embedding_profiles.c.model)
        .order_by(embedding_profiles.c.dims, models.c.parameters, embedding_profiles.c.profile)
    )


def _model(row: Row[Any]) -> tuple[str, EmbeddingModel]:
    (
        profile,
        name,
        dims,
        query_prefix,
        document_prefix,
        matryoshka,
        chunk,
        passage,
        weak_match,
        answered_match,
        same_topic,
    ) = row
    return profile, EmbeddingModel(
        name,
        dims,
        duplicate=None if chunk is None else DuplicateCosine(chunk, passage),
        profile=profile,
        weak_match=weak_match,
        answered_match=answered_match,
        same_topic=same_topic,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        matryoshka=bool(matryoshka),
    )


async def _records(statement: Select[Any]) -> list[dict[str, Any]]:
    """The rows of `statement` keyed by column name, so they convert to a struct by field name."""
    async with db.read() as conn:
        return [db.record(row) for row in await conn.execute(statement)]


# each database file's profiles are read once, since every search and indexing step resolves its
# model here: a process sees gap bars `calibrate_gaps --write` stores only after a restart
_embedders: dict[Path, dict[str, EmbeddingModel]] = {}


async def embedders() -> dict[str, EmbeddingModel]:
    """Every embedding profile's model, in picker order. "none" is not among them: full-text
    only is the absence of a model."""
    cached = _embedders.get(home.DB_FILE)
    if cached is None:
        async with db.read() as conn:
            rows = await conn.execute(_profiles(*_PROFILE))
            cached = _embedders[home.DB_FILE] = dict(map(_model, rows))
    return cached


def _metadata[M: ModelMetadata](record: dict[str, Any], kind: type[M]) -> M:
    """A record of `_METADATA`, with what the loaders say about its model. The conversion checks
    each column's type, so a malformed `released` fails here, not in a client."""
    name = record["name"]
    hosting = {"runtime": hardware.runtime(name), "devices": hardware.devices(name)}
    return msgspec.convert(record | hosting, kind)


async def embedding_metadata() -> dict[str, EmbedderMetadata]:
    """Every embedding profile's metadata, in picker order: its model's, under the profile's own
    description where it has one (one model cut two ways is two profiles)."""
    records = await _records(
        _profiles(
            embedding_profiles.c.profile,
            func.coalesce(embedding_profiles.c.description, models.c.description).label(
                "description"
            ),
            *_METADATA,
            embedding_profiles.c.dims.label("dimensions"),
        )
    )
    return {record["profile"]: _metadata(record, EmbedderMetadata) for record in records}


async def rerankers() -> dict[str, RerankerMetadata]:
    """Every reranker model's metadata, smallest first: the first is the smallest there is."""
    records = await _records(
        select(models.c.description, *_METADATA)
        .where(models.c.kind == "reranker")
        .order_by(models.c.parameters, models.c.name)
    )
    return {record["name"]: _metadata(record, RerankerMetadata) for record in records}


async def calibration(model: str) -> RerankerCalibration:
    """How `model`'s scores read; the uncalibrated defaults for a model the catalogue has no row
    for, as a reranker added after the seed has."""
    records = await _records(
        select(*db.columns_of(reranker_calibration, RerankerCalibration)).where(
            reranker_calibration.c.model == model
        )
    )
    return msgspec.convert(records[0], RerankerCalibration) if records else UNCALIBRATED


async def embedding_model(settings: UserSettings) -> EmbeddingModel | None:
    """The model `settings` embed with, on the hardware they choose; None for full-text only."""
    if settings.embedding == NO_EMBEDDING:
        return None
    model = (await embedders()).get(settings.embedding)
    if model is None:
        raise InvalidInput(f"unknown embedding profile: {settings.embedding}")
    return msgspec.structs.replace(model, accelerator=settings.pipeline.accelerator)


async def unknown(conn: AsyncConnection, settings: UserSettings | CollectionOverrides) -> str:
    """What `settings` name that the catalogue does not hold, as one message; empty when nothing.
    On the caller's connection: the settings load runs it with the read it checks."""
    missing: list[str] = []
    profile = settings.embedding if isinstance(settings, UserSettings) else NO_EMBEDDING
    if profile != NO_EMBEDDING and not await _exists(
        conn, select(embedding_profiles.c.profile).where(embedding_profiles.c.profile == profile)
    ):
        missing.append(f"unknown embedding profile: {profile}")
    reranker = settings.search.reranker_model
    if reranker is not None and not await _exists(
        conn,
        select(models.c.name).where(models.c.name == reranker, models.c.kind == "reranker"),
    ):
        missing.append(f"unknown reranker model: {reranker}")
    return "; ".join(missing)


async def _exists(conn: AsyncConnection, statement: Select[Any]) -> bool:
    return await conn.scalar(statement) is not None


async def check(settings: UserSettings | CollectionOverrides) -> None:
    """Reject settings that name a profile or a reranker model the catalogue does not hold, or
    user settings whose hardware setting leaves a model they use nowhere to run, the describer
    of `Descriptors.LLM` among them. Only here, where
    settings are written: a stored row that no longer runs still loads, and its model reports why.
    """
    async with db.read() as conn:
        problem = await unknown(conn, settings)
    if not problem and isinstance(settings, UserSettings):
        problem = await _stranded(settings)
    if problem:
        raise InvalidInput(problem)


async def _stranded(settings: UserSettings) -> str:
    """The models `settings` use that `hardware.device` finds no device for, as one message."""
    accelerator = settings.pipeline.accelerator
    used = []
    if settings.embedding != NO_EMBEDDING:
        used.append((await embedders())[settings.embedding].name)
    if settings.search.reranker != Reranker.NONE:
        used.append(settings.search.reranker_model)
    if describer := gguf_models.describer(settings.pipeline):
        used.append(describer)
    return "; ".join(
        hardware.nowhere(name) for name in used if hardware.device(name, accelerator) is None
    )
