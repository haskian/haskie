"""The model catalogue: every model the runtimes can load, its metadata, and the embedding
profiles a user picks from. It lives in the database (`models`, `embedding_profiles`), seeded once
from `seed.sql`, so this module reads it and holds none of it.

How a model loads stays in code: the pinned revisions in `embed`, `onnx_rerank` and `mlx_models`
name reviewed code and weights, and a row cannot add a loader. Settings name a profile and a
reranker model by key, and those keys are checked here, because a settings struct decodes without
the database: at the write boundaries (`check`), and when the stored row is read (`unknown`).
"""

import hashlib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import aiosqlite
import msgspec

from haskie import db, home
from haskie.errors import InvalidInput
from haskie.indexing import hardware
from haskie.indexing.hardware import Device, Runtime
from haskie.settings import NO_EMBEDDING, Accelerator, CollectionOverrides, UserSettings


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


class DuplicateCosine(msgspec.Struct, frozen=True):
    """The raw cosines two search results must exceed to count as the same point
    (`search.collapse`). Exceeded, not reached: Set-Encoder's near-duplicate is Jaccard > 0.5."""

    chunk: float  # chunk to chunk, and containment
    passage: float  # mean vector to mean vector: means are smoother, so they run higher


class Matryoshka(msgspec.Struct, frozen=True):
    """A model trained so that the first values of its vector are a vector of their own
    (Matryoshka Representation Learning): its vectors are cut to `EmbeddingModel.dims`, then
    normalized again, which makes the index and every comparison smaller for a small loss."""

    layer_norm: bool = False  # nomic's recipe: layer-normalize the whole vector before the cut


class EmbeddingModel(msgspec.Struct):
    name: str
    dims: int  # of the vectors stored and searched: the model's own, or its Matryoshka cut
    accelerator: Accelerator = Accelerator.AUTO
    duplicate: DuplicateCosine | None = None  # None: search results are compared by words alone
    # What the model was trained to read ahead of a query and of a passage (e5's "query: ",
    # nomic's "search_query: "); empty for models that need none. They shape every vector, so
    # changing one is changing the model. The document prefix is part of `cache_name`, so the
    # cached vectors it shaped are not served after it changes.
    query_prefix: str = ""
    document_prefix: str = ""
    matryoshka: Matryoshka | None = None

    @property
    def cache_name(self) -> str:
        """What the embedding cache keys this model's vectors by: everything that shapes a stored
        vector. Its name and size are readable; the document prefix and the Matryoshka recipe
        are hashed. The query prefix, the accelerator and the thresholds shape no stored vector,
        so changing them keeps the cache."""
        shaping = msgspec.json.encode([self.document_prefix, self.matryoshka])
        return f"{self.name}@{self.dims}:{hashlib.sha256(shaping).hexdigest()[:12]}"


# the columns of `models` a `ModelMetadata` reads but the description, which a profile may override
_METADATA = (
    "m.name, m.parameters, m.context_tokens, m.languages, m.license, m.released, m.model_card_url"
)
_PROFILE = (
    "p.profile, p.model, p.dims, p.query_prefix, p.document_prefix, p.matryoshka_layer_norm, "
    "p.duplicate_chunk, p.duplicate_passage"
)
# the order every picker lists them in: the shorter vectors first, and at one size the smaller
# model first; the profile key breaks a tie, so the order never depends on the insert order
_PROFILE_ORDER = "order by p.dims, m.parameters, p.profile"
_PROFILES_FROM = "from embedding_profiles p join models m on m.name = p.model "


def _model(row: Any) -> tuple[str, EmbeddingModel]:
    profile, name, dims, query_prefix, document_prefix, layer_norm, chunk, passage = row
    return profile, EmbeddingModel(
        name,
        dims,
        duplicate=None if chunk is None else DuplicateCosine(chunk, passage),
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        matryoshka=None if layer_norm is None else Matryoshka(layer_norm=bool(layer_norm)),
    )


async def _records(sql: str) -> list[dict[str, Any]]:
    """The rows of `sql` keyed by column name, so they convert to a struct by field name."""
    async with db.connect() as conn:
        cursor = await conn.execute(sql)
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in await cursor.fetchall()]


# nothing writes the catalogue after the seed, so each database file's profiles are read once:
# every search and indexing step resolves its model here
_embedders: dict[Path, dict[str, EmbeddingModel]] = {}


async def embedders() -> dict[str, EmbeddingModel]:
    """Every embedding profile's model, in picker order. "none" is not among them: full-text
    only is the absence of a model."""
    cached = _embedders.get(home.DB_FILE)
    if cached is None:
        async with db.connect() as conn:
            cursor = await conn.execute(f"select {_PROFILE} {_PROFILES_FROM}{_PROFILE_ORDER}")
            cached = _embedders[home.DB_FILE] = dict(map(_model, await cursor.fetchall()))
    return cached


def _metadata[M: ModelMetadata](
    record: dict[str, Any], kind: type[M], devices: Callable[[str], tuple[Device, ...]]
) -> M:
    """A record of `_METADATA`, with what the loaders say about its model. The conversion checks
    each column's type, so a malformed `released` fails here, not in a client."""
    name = record["name"]
    hosting = {"runtime": hardware.runtime(name), "devices": devices(name)}
    return msgspec.convert(record | hosting, kind)


async def embedding_metadata() -> dict[str, EmbedderMetadata]:
    """Every embedding profile's metadata, in picker order: its model's, under the profile's own
    description where it has one (one model cut two ways is two profiles)."""
    records = await _records(
        f"select p.profile, coalesce(p.description, m.description) as description, {_METADATA}, "
        f"p.dims as dimensions {_PROFILES_FROM}{_PROFILE_ORDER}"
    )
    return {
        record["profile"]: _metadata(record, EmbedderMetadata, hardware.embedder_devices)
        for record in records
    }


async def rerankers() -> dict[str, RerankerMetadata]:
    """Every reranker model's metadata, smallest first: the first is the smallest there is."""
    records = await _records(
        f"select m.description, {_METADATA} from models m where m.kind = 'reranker' "
        "order by m.parameters, m.name"
    )
    return {
        record["name"]: _metadata(record, RerankerMetadata, hardware.reranker_devices)
        for record in records
    }


async def embedding_model(settings: UserSettings) -> EmbeddingModel | None:
    """The model `settings` embed with, on the hardware they choose; None for full-text only."""
    if settings.embedding == NO_EMBEDDING:
        return None
    model = (await embedders()).get(settings.embedding)
    if model is None:
        raise InvalidInput(f"unknown embedding profile: {settings.embedding}")
    return msgspec.structs.replace(model, accelerator=settings.pipeline.accelerator)


async def unknown(conn: aiosqlite.Connection, settings: UserSettings | CollectionOverrides) -> str:
    """What `settings` name that the catalogue does not hold, as one message; empty when nothing.
    On the caller's connection: the settings load runs it with the read it checks."""
    missing: list[str] = []
    profile = settings.embedding if isinstance(settings, UserSettings) else NO_EMBEDDING
    if profile != NO_EMBEDDING and not await _exists(
        conn, "select 1 from embedding_profiles where profile = ?", profile
    ):
        missing.append(f"unknown embedding profile: {profile}")
    reranker = settings.search.reranker_model
    if reranker is not None and not await _exists(
        conn, "select 1 from models where name = ? and kind = 'reranker'", reranker
    ):
        missing.append(f"unknown reranker model: {reranker}")
    return "; ".join(missing)


async def _exists(conn: aiosqlite.Connection, sql: str, key: str) -> bool:
    cursor = await conn.execute(sql, (key,))
    return await cursor.fetchone() is not None


async def check(settings: UserSettings | CollectionOverrides) -> None:
    """Reject settings that name a profile or a reranker model the catalogue does not hold."""
    async with db.connect() as conn:
        problem = await unknown(conn, settings)
    if problem:
        raise InvalidInput(problem)
