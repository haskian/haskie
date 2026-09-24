"""Search over a live index: one collection, and a session that fans out over several.

Both need a real index, which the pipeline writes, so every test here takes the `dbos` fixture
and imports its documents through it. The session tests assert the mechanism - one query
embedding, one model check, one cross-encoder pass over the merge - and not only the end state.
"""

import shutil
from pathlib import Path

import pytest
from conftest import (
    attach_document,
    events,
    import_document,
    legacy_index,
    one_part,
)

from haskie import db, home
from haskie.collection.collection import Collection
from haskie.errors import NotFound
from haskie.indexing import chunk, embed, models
from haskie.search import flow, retrieval, session
from haskie.settings import (
    ChunkSettings,
    CollectionOverrides,
    EmbeddingProfile,
    Reranker,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    save_user_settings,
)

pytestmark = pytest.mark.anyio


async def test_search_limit_user_and_collection_level(dbos, tmp_path: Path) -> None:
    await save_user_settings(UserSettings(search=SearchSettings(limit=2)))
    collection = await Collection.create("lim")
    await collection.set_overrides(CollectionOverrides(chunk_size=30))
    body = "".join(f"# H{i}\n\ncommon token {i}\n\n" for i in range(6))
    doc = await import_document(dbos, "m.md", body, tmp_path)
    await attach_document(dbos, "lim", doc.name)

    assert len(await collection.search("common")) == 2, "user default"
    await collection.set_overrides(
        CollectionOverrides(chunk_size=30, search=SearchOverrides(limit=4))
    )
    assert (await collection.info()).search.limit == 4
    assert len(await collection.search("common")) == 4, "collection override"
    assert len(await collection.search("common", SearchOverrides(limit=1))) == 1, (
        "explicit beats both"
    )
    effective = (await collection.info()).effective
    assert effective.chunk_size == 30, "a search override does not leak into the chunk settings"

    await session.set_collections("s", ["lim"])
    chosen = await session.collections_for("s")
    assert len(await flow.chunks(chosen, "common")) == 2, "session cut to the user limit"
    assert len(await flow.chunks(chosen, "common", limit=3)) == 3


async def test_an_outdated_index_is_reported_and_rebuilt(dbos, tmp_path: Path) -> None:
    """A table this build cannot read is reported rather than dropped on sight; the write
    path is what replaces it, out of the document's embedding cache."""
    collection = await Collection.create("old")
    doc = await import_document(dbos, "g.md", "# Hi\n\nhello world\n", tmp_path)
    collection.index_dir.mkdir(parents=True)
    old = legacy_index(collection.index_dir, "g.md", "hello world", heading="Hi")

    assert (await collection.info()).index_outdated is True
    assert old.count_rows() == 1, "and left alone until something writes"

    await attach_document(dbos, "old", doc.name)  # the write path drops the old table and rebuilds

    assert (await collection.info()).index_outdated is False
    (hit,) = await collection.search("hello")
    assert hit.source_path == doc.relative(doc.original) and hit.line_start == 3
    assert hit.source_file == str(home.HOME / hit.source_path), "absolute, for a tool outside"


async def test_home_is_portable(dbos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Move the whole home directory: DB, files and index keep working, paths still resolve."""
    await Collection.create("port")
    doc = await import_document(dbos, "p.md", "# P\n\nportable text\n", tmp_path)
    await attach_document(dbos, "port", doc.name)

    moved = tmp_path / "elsewhere"
    shutil.copytree(home.HOME, moved)
    monkeypatch.setattr(home, "HOME", moved)  # every other path is derived from it
    monkeypatch.setattr(db, "_migrated", set())

    again = await Collection.get("port")
    (hit,) = await again.search("portable")
    assert hit.source_file == str(moved / hit.source_path), "resolved against the home it is in"
    assert (moved / hit.source_path).read_bytes().startswith(b"# P")
    assert (moved / hit.markdown_path).exists()


# --- cross-collection session search -----------------------------------------------


async def test_session_search_merges_collections(dbos, tmp_path: Path) -> None:
    for name in ("a", "b"):
        await Collection.create(name)
        body = f"# {name}\n\nshared token {name}\n"
        doc = await import_document(dbos, f"{name}.md", body, tmp_path)
        await attach_document(dbos, name, doc.name)
    with pytest.raises(NotFound, match="collection not found"):
        await session.set_collections("s1", ["a", "ghost"])
    await session.set_collections("s1", ["a", "b"])

    hits = await flow.chunks(await session.collections_for("s1"), "shared", limit=10)

    assert {h.collection for h in hits} == {"a", "b"}
    unknown = await session.collections_for("unknown-session")
    assert await flow.chunks(unknown, "shared") == []


async def test_session_search_counts_a_shared_document_once(dbos, tmp_path: Path) -> None:
    """A document in two chosen collections sits in both their tables, so the same passage comes
    back twice. A session search is about passages, so it is merged once and credited to the
    first collection of the session that returned it."""
    for name in ("first", "second"):
        await Collection.create(name)
    shared = await import_document(dbos, "shared.md", "# S\n\nshared token here\n", tmp_path)
    solo = await import_document(dbos, "solo.md", "# O\n\nshared token too\n", tmp_path)
    for name in ("first", "second"):
        await attach_document(dbos, name, shared.name)
    await attach_document(dbos, "first", solo.name)
    await session.set_collections("s", ["first", "second"])

    hits = await flow.chunks(await session.collections_for("s"), "shared", limit=10)

    keys = [(h.document, h.seq) for h in hits]
    assert len(keys) == len(set(keys)), f"one hit per passage, got {keys}"
    assert sorted(h.document for h in hits) == sorted([solo.name, shared.name])
    assert {h.collection for h in hits} == {"first"}, "credited to the first that returned it"
    assert {h.collection for h in await Collection("second").search("shared")} == {"second"}


async def test_session_search_embeds_once_and_checks_the_model_once(dbos, monkeypatch) -> None:
    """Three collections, one embedding: the query used to be embedded (and the model checked)
    once per collection."""
    from haskie.collection.index import Row
    from haskie.settings import PROFILES

    await save_user_settings(UserSettings(embedding=EmbeddingProfile.COMPACT))
    compact = PROFILES[EmbeddingProfile.COMPACT]
    assert compact is not None
    for name in ("a", "b", "c"):
        collection = await Collection.create(name)
        (chunk_,) = chunk.split(f"# {name}\n\nshared token {name}\n", ChunkSettings())
        index = collection.index_with(compact)
        # a vector of its own per collection: three equal vectors would be one point, folded
        vector = [0.1] * compact.dims
        vector["abc".index(name)] = 1.0
        row = Row(chunk=chunk_, vector=vector, seq=1)
        # a document name per collection: the merge keys on (document, seq), so one name
        # shared by all three would be one passage and this test would see a single hit
        await index.add_parts(
            f"{name}.md", f"documents/{name}.md", f"documents/{name}.md.md", one_part(0, [row])
        )
        await index.finish()
    await session.set_collections("s", ["a", "b", "c"])
    embedded: list[str] = []
    checked: list[tuple[str, str]] = []

    def fake_embed(model, text: str) -> list[float]:
        embedded.append(text)
        return [0.1] * model.dims

    async def require_ready(kind: str, model: str) -> None:
        checked.append((kind, model))

    monkeypatch.setattr(retrieval, "embed_query", fake_embed)
    monkeypatch.setattr(models, "require_ready", require_ready)

    hits = await flow.chunks(await session.collections_for("s"), "shared", limit=10)

    assert {h.collection for h in hits} == {"a", "b", "c"}
    assert embedded == ["shared"], "one embedding for the whole fan-out"
    assert checked == [("embedding", compact.name)], "one check, not one per collection"


async def test_session_search_folds_a_near_duplicate_by_its_vector(dbos, monkeypatch) -> None:
    """The fold runs on the vectors the index returned: two documents whose chunks embed alike
    come back as one hit that names the other, though their words have little in common. If the
    rows lost their vectors, the search would fall back to words and keep both."""
    from haskie.collection.index import Row
    from haskie.settings import PROFILES

    await save_user_settings(UserSettings(embedding=EmbeddingProfile.COMPACT))
    compact = PROFILES[EmbeddingProfile.COMPACT]
    assert compact is not None
    vector = [0.1] * compact.dims  # one point, embedded twice
    bodies = {
        "a": "# Retries\n\nA retried call has to be idempotent, or it happens twice.\n",
        "b": "# Delivery\n\nMake each request safe to repeat: the job may send it again.\n",
    }
    for name, body in bodies.items():
        collection = await Collection.create(name)
        (chunk_,) = chunk.split(body, ChunkSettings())
        index = collection.index_with(compact)
        await index.add_parts(
            f"{name}.md",
            f"documents/{name}.md",
            f"documents/{name}.md.md",
            one_part(0, [Row(chunk=chunk_, vector=vector, seq=1)]),
        )
        await index.finish()
    await session.set_collections("s", ["a", "b"])

    async def require_ready(kind: str, model: str) -> None:
        return None

    monkeypatch.setattr(retrieval, "embed_query", lambda model, text: vector)
    monkeypatch.setattr(models, "require_ready", require_ready)

    (hit,) = await flow.chunks(await session.collections_for("s"), "idempotent retries", limit=10)

    assert [ref.document for ref in hit.also_in] == [({"a.md", "b.md"} - {hit.document}).pop()], (
        "the other document, folded in"
    )
    assert hit.also_count == 1
    assert hit.also_in[0].similarity == pytest.approx(1.0), "by its vector: the words differ"


async def test_session_search_reranks_once_over_the_merge(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """The cross-encoder sees the merged candidates of every collection once, and its score is
    the score of the returned hits."""
    await save_user_settings(
        UserSettings(search=SearchSettings(limit=2, candidates=4, reranker=Reranker.CROSS_ENCODER))
    )
    for name, token in (("a", "alpha"), ("b", "beta")):
        collection = await Collection.create(name)
        await collection.set_overrides(CollectionOverrides(chunk_size=30))
        body = "".join(f"# H{i}\n\nshared {token} {i}\n\n" for i in range(3))
        doc = await import_document(dbos, f"{name}.md", body, tmp_path)
        await attach_document(dbos, name, doc.name)
    await session.set_collections("s", ["a", "b"])
    calls: list[list[str]] = []

    async def require_ready(kind: str, model: str) -> None:
        return None

    monkeypatch.setattr(models, "require_ready", require_ready)

    def fake_rerank(model: str, query: str, texts: list[str]) -> list[float]:
        calls.append(texts)
        return [float(i) for i in range(len(texts))]

    monkeypatch.setattr(embed, "rerank_scores", fake_rerank)

    hits = await flow.chunks(await session.collections_for("s"), "shared")

    assert len(calls) == 1, "one cross-encoder pass, not one per collection"
    (texts,) = calls
    assert len(texts) == 4, "the merge is cut to `candidates` before it is rescored"
    assert any("alpha" in t for t in texts), "candidates from both collections"
    assert any("beta" in t for t in texts)
    # the cross-encoder reads each chunk under its heading path (`chunk.framed`)
    read = [chunk.framed(h.frame, h.text) for h in hits]
    assert read == [texts[-1], texts[-2]], "best cross-encoder score first"
    assert [h.score for h in hits] == [3.0, 2.0], "the hit carries the cross-encoder score"


async def test_session_search_propagates_a_broken_collection(
    dbos, tmp_path: Path, monkeypatch, caplog
) -> None:
    """A collection that cannot answer must not be silently dropped: an empty result reads as
    "no match", which is a different answer."""
    from haskie.collection.index import CollectionIndex

    for name in ("a", "b"):
        await Collection.create(name)
        doc = await import_document(dbos, f"{name}.md", f"# {name}\n\nshared token\n", tmp_path)
        await attach_document(dbos, name, doc.name)
    await session.set_collections("s", ["a", "b"])

    async def boom(self, query, vector, settings_, limit):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(CollectionIndex, "search_rows", boom)

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="index unreadable"):
            await flow.chunks(await session.collections_for("s"), "shared")

    assert "session_collection_search_failed" in events(caplog)
