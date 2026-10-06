"""One collection's controlled vocabulary on disk: a LanceDB database beside its index, in four
tables, written by the vocabulary run (`workflows.build_vocabulary`) alone, one run per collection
at a time. What a preferred term is, and how variants find theirs, is `sections.vocabulary`.

- `descriptors`: every descriptor of every indexed member's sections, with the section it
  describes and its variant. Rewritten whole by each run (`collect`): the members' cache entries
  hold the descriptors themselves, shared by every collection that chunks a document alike, so
  this table is this collection's view of them, never the only copy.
- `vectors`: each variant's vector (`sections.vocabulary.INSTRUCTION`). Kept across runs: a run
  embeds only the variants a new member brought (`embed_missing`).
- `verdicts`: each pair of neighbours the describer is asked about, and its answer, the mean
  P(yes) of both orders; null until asked. Kept across runs too: a run asks only the new pairs
  (`plan`, `judge_missing`).
- `terms`: the preferred terms, each with its variants and its vector (`build`). What the
  collection's search reads (`preferred`).

A collection's folder holds this database, so a rename moves it and a delete removes it.
"""

import asyncio
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import anyio
import lancedb
import numpy as np
import pyarrow as pa

from haskie.collection.collection import Collection
from haskie.collection.index import vector_field
from haskie.errors import NotFound
from haskie.indexing import embed_cache
from haskie.logs import get_logger
from haskie.sections import vocabulary
from haskie.sections.vocabulary import Pair

_log = get_logger(__name__)

DESCRIPTORS = "descriptors"
VECTORS = "vectors"
VERDICTS = "verdicts"
TERMS = "terms"
MODEL_KEY = b"haskie.model"  # the vectors table's metadata: which model made its vectors
BUILD_KEY = b"haskie.vocabulary-build"  # unique across collections, renames and rebuilds

type Embed = Callable[[list[str]], list[list[float]]]  # one vector per text
type Judge = Callable[[str], float]  # P(yes) of the answer to one prompt

_DESCRIPTORS = pa.schema(
    [pa.field(name, pa.string()) for name in ("document_id", "section_id", "descriptor", "variant")]
)
_VERDICTS = pa.schema(
    [pa.field("a", pa.string()), pa.field("b", pa.string()), pa.field("same", pa.float32())]
)

# Each build identifies its contents even when another table takes the same collection name.
_preferred: dict[str, tuple[bytes, dict[str, str]]] = {}


def path(collection: str) -> Path:
    return Collection(collection).root / "vocabulary"


async def _connect(collection: str, create: bool) -> lancedb.AsyncConnection | None:
    """The collection's vocabulary database; None when it has none and `create` is off: a read
    never creates one (`connect_async` alone would create the folder)."""
    where = path(collection)
    if not create and not await anyio.Path(where).exists():
        return None
    return await lancedb.connect_async(str(where))


async def _table(conn: lancedb.AsyncConnection, name: str) -> lancedb.AsyncTable | None:
    if name not in (await conn.list_tables()).tables:
        return None
    return await conn.open_table(name)


async def _read(table: lancedb.AsyncTable | None, columns: list[str]) -> pa.Table | None:
    if table is None:
        return None
    return await table.query().select(columns).to_arrow()


async def collect(collection: str) -> int:
    """Rewrite the `descriptors` table from the indexed members' sections; returns how many
    variants they hold. 0 for a collection that is gone, or whose members say nothing."""
    try:
        found = await Collection.get(collection)
    except NotFound:  # deleted while the run waited, which is no failure of it
        return 0
    caches = await found.indexed_caches()
    docs = sorted(caches)
    sections = await asyncio.gather(*(embed_cache.read_sections(doc, caches[doc]) for doc in docs))
    rows: dict[str, list[str]] = {name: [] for name in _DESCRIPTORS.names}
    for doc, described in zip(docs, sections, strict=True):
        for section in described:
            for phrase in section.descriptors:
                rows["document_id"].append(doc)
                rows["section_id"].append(section.id)
                rows["descriptor"].append(phrase)
                rows["variant"].append(vocabulary.variant(phrase))
    conn = await _connect(collection, create=True)
    assert conn is not None
    data = pa.table(rows, schema=_DESCRIPTORS)
    await conn.create_table(DESCRIPTORS, data=data, schema=_DESCRIPTORS, mode="overwrite")
    return len(set(rows["variant"]))


async def _variants(conn: lancedb.AsyncConnection) -> tuple[list[str], list[int], list[str]]:
    """The collection's variants in sorted order, with how often its sections use each and the
    form they write it in most often (ties: the first in sorted order)."""
    found = await _read(await _table(conn, DESCRIPTORS), ["descriptor", "variant"])
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    if found is not None:
        for phrase, key in zip(
            found["descriptor"].to_pylist(), found["variant"].to_pylist(), strict=True
        ):
            forms[key][phrase.strip()] += 1
    variants = sorted(forms)
    uses = [forms[one].total() for one in variants]
    shown = [min(forms[one].items(), key=lambda item: (-item[1], item[0]))[0] for one in variants]
    return variants, uses, shown


async def _vectors(conn: lancedb.AsyncConnection, model: str) -> lancedb.AsyncTable | None:
    """The vectors table, dropped first when another model made its vectors."""
    table = await _table(conn, VECTORS)
    if table is not None and ((await table.schema()).metadata or {}).get(MODEL_KEY) != (
        model.encode()
    ):
        await conn.drop_table(VECTORS)
        await conn.drop_table(VERDICTS, ignore_missing=True)  # judged pairs of other neighbours
        table = None
    return table


async def embed_missing(collection: str, model: str, embed: Embed, limit: int) -> int:
    """Embed up to `limit` variants the vectors table lacks, by `model`; returns how many, 0
    once it holds every one."""
    conn = await _connect(collection, create=True)
    assert conn is not None
    variants, _, _ = await _variants(conn)
    table = await _vectors(conn, model)
    known = await _read(table, ["variant"])
    have = set() if known is None else set(known["variant"].to_pylist())
    missing = [one for one in variants if one not in have][:limit]
    if not missing:
        return 0
    vectors = await anyio.to_thread.run_sync(embed, missing)
    data = pa.table(
        {
            "variant": pa.array(missing, pa.string()),
            "vector": pa.array(vectors, pa.list_(pa.float32(), len(vectors[0]))),
        }
    )
    if table is None:
        schema = pa.schema(
            [pa.field("variant", pa.string()), vector_field(len(vectors[0]))],
            metadata={MODEL_KEY: model.encode()},
        )
        await conn.create_table(VECTORS, data=data.cast(schema), schema=schema)
    else:
        await table.add(data.cast(await table.schema()))
    return len(missing)


async def _matrix(
    conn: lancedb.AsyncConnection, model: str
) -> tuple[list[str], list[int], list[str], np.ndarray]:
    """The variants, their uses and forms, and their vectors in the same order, unit length."""
    variants, uses, shown = await _variants(conn)
    found = await _read(await _vectors(conn, model), ["variant", "vector"])
    if found is None:
        return variants, uses, shown, np.empty((len(variants), 0), dtype=np.float32)
    rows = dict(zip(found["variant"].to_pylist(), found["vector"].to_numpy(), strict=True))
    matrix = np.stack([rows[one] for one in variants]).astype(np.float32) if variants else None
    if matrix is None:
        return variants, uses, shown, np.empty((0, 0), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return variants, uses, shown, matrix


async def plan(collection: str, model: str) -> int:
    """Add the pairs `sections.vocabulary.to_judge` finds, and the verdicts table lacks, as yet
    unasked; returns how many it added."""
    conn = await _connect(collection, create=True)
    assert conn is not None
    variants, _, _, vectors = await _matrix(conn, model)
    pairs = await anyio.to_thread.run_sync(vocabulary.to_judge, variants, vectors)
    table = await _table(conn, VERDICTS)
    known = await _read(table, ["a", "b"])
    asked = (
        set()
        if known is None
        else set(zip(known["a"].to_pylist(), known["b"].to_pylist(), strict=True))
    )
    new = [one for one in pairs if one not in asked]
    data = pa.table(
        {
            "a": [a for a, _ in new],
            "b": [b for _, b in new],
            "same": pa.nulls(len(new), pa.float32()),
        },
        schema=_VERDICTS,
    )
    if table is None:
        await conn.create_table(VERDICTS, data=data, schema=_VERDICTS)
    elif new:
        await table.add(data)
    return len(new)


async def judge_missing(collection: str, judge: Judge, limit: int) -> int:
    """Ask the describer about up to `limit` pairs not yet asked, both orders each; returns how
    many, 0 once every pair has its verdict."""
    conn = await _connect(collection, create=True)
    assert conn is not None
    table = await _table(conn, VERDICTS)
    if table is None:
        return 0
    found = await table.query().where("same IS NULL").select(["a", "b"]).limit(limit).to_arrow()
    pairs = list(zip(found["a"].to_pylist(), found["b"].to_pylist(), strict=True))
    if not pairs:
        return 0

    def ask(a: str, b: str) -> float:
        one = judge(vocabulary.JUDGE_PROMPT.format(a=a, b=b))
        return (one + judge(vocabulary.JUDGE_PROMPT.format(a=b, b=a))) / 2

    same = [await anyio.to_thread.run_sync(ask, a, b) for a, b in pairs]
    data = pa.table(
        {"a": [a for a, _ in pairs], "b": [b for _, b in pairs], "same": same}, schema=_VERDICTS
    )
    await table.merge_insert(["a", "b"]).when_matched_update_all().execute(data)
    return len(pairs)


async def _verdicts(conn: lancedb.AsyncConnection) -> dict[Pair, float]:
    found = await _read(await _table(conn, VERDICTS), ["a", "b", "same"])
    if found is None:
        return {}
    return {
        (a, b): same
        for a, b, same in zip(
            found["a"].to_pylist(), found["b"].to_pylist(), found["same"].to_pylist(), strict=True
        )
        if same is not None
    }


async def build(collection: str, model: str) -> int:
    """Cluster the variants into preferred terms and rewrite the `terms` table, then compact the
    tables the run appended to; returns how many terms."""
    conn = await _connect(collection, create=True)
    assert conn is not None
    variants, uses, shown, vectors = await _matrix(conn, model)
    verdicts = await _verdicts(conn)
    preferred = await anyio.to_thread.run_sync(
        vocabulary.cluster, variants, uses, vectors, verdicts
    )
    found = vocabulary.terms(variants, uses, shown, preferred)
    at = {one: index for index, one in enumerate(variants)}
    dims = vectors.shape[1] if len(vectors) else 1
    data = pa.table(
        {
            "variant": [one.variant for one in found],
            "term": [one.term for one in found],
            "uses": pa.array([one.uses for one in found], pa.int64()),
            "variants": pa.array([one.variants for one in found], pa.list_(pa.string())),
            "vector": pa.array(
                [vectors[at[one.variant]].tolist() for one in found], pa.list_(pa.float32(), dims)
            ),
        }
    )
    data = data.replace_schema_metadata({BUILD_KEY: uuid4().hex.encode()})
    await conn.create_table(TERMS, data=data, mode="overwrite")
    for name in (VECTORS, VERDICTS):
        if (table := await _table(conn, name)) is not None:
            await table.optimize()
    _log.info("vocabulary_built", collection=collection, variants=len(variants), terms=len(found))
    return len(found)


async def preferred(collection: str) -> dict[str, str]:
    """Each variant of the collection's vocabulary, and the term it stands for; empty before its
    first run. Read once per build; older tables without a build identity are read uncached."""
    conn = await _connect(collection, create=False)
    table = None if conn is None else await _table(conn, TERMS)
    if table is None:
        return {}
    identity = ((await table.schema()).metadata or {}).get(BUILD_KEY)
    cached = _preferred.get(collection)
    if identity is not None and cached is not None and cached[0] == identity:
        return cached[1]
    found = await _read(table, ["term", "variants"])
    assert found is not None
    mapping = {
        one: term
        for term, variants in zip(
            found["term"].to_pylist(), found["variants"].to_pylist(), strict=True
        )
        for one in variants
    }
    if identity is not None:
        _preferred[collection] = (identity, mapping)
    return mapping
