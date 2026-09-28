"""Search over a live index: one collection, and a session that fans out over several.

Both need a real index, which the pipeline writes, so every test here takes the `dbos` fixture
and imports its documents through it. The session tests assert the mechanism - one query
embedding, one model check, one cross-encoder pass over the merge - and not only the end state.
"""

import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    attach_document,
    collection_hits,
    events,
    import_document,
    legacy_index,
    one_part,
)

from haskie import db, home
from haskie.catalogue import catalogue
from haskie.collection.collection import Collection
from haskie.errors import NotFound
from haskie.indexing import chunk, embed, models
from haskie.search import aspects, flow, retrieval, session
from haskie.settings import (
    ChunkSettings,
    CollectionOverrides,
    Reranker,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    save_user_settings,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def models_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every model counts as loaded: these searches stub what the models answer."""

    async def ready(kind: str, model: str) -> None:
        return None

    monkeypatch.setattr(models, "require_ready", ready)


async def test_search_limit_user_and_collection_level(dbos, tmp_path: Path) -> None:
    """The call's limit, else the collection's when it is the one collection in scope, else the
    user's: the one search route reads it the way it reads every other collection setting."""
    await save_user_settings(UserSettings(search=SearchSettings(limit=2)))
    collection = await Collection.create("lim")
    await Collection.create("other")
    await collection.set_overrides(CollectionOverrides(chunk_size=30))
    body = "".join(f"# H{i}\n\ncommon token {i}\n\n" for i in range(6))
    doc = await import_document(dbos, "m.md", body, tmp_path)
    await attach_document(dbos, "lim", doc.name)

    assert len(await flow.chunks(["lim"], "common")) == 2, "user default"
    await collection.set_overrides(
        CollectionOverrides(chunk_size=30, search=SearchOverrides(limit=4))
    )
    assert (await collection.info()).search.limit == 4
    assert len(await flow.chunks(["lim"], "common")) == 4, "collection override"
    assert len(await flow.chunks(["lim"], "common", limit=1)) == 1, "explicit beats both"
    assert len(await flow.chunks(["lim", "other"], "common")) == 2, (
        "several collections in scope: the user's"
    )
    effective = (await collection.info()).effective
    assert effective.chunk_size == 30, "a search override does not leak into the chunk settings"

    await session.set_collections("s", ["lim"])
    chosen = await session.collections_for("s")
    assert len(await flow.chunks(chosen, "common")) == 4, "a session of one: the collection's"
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
    (hit,) = await collection_hits(collection.name, "hello")
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
    monkeypatch.setattr(db, "_engines", {})

    again = await Collection.get("port")
    (hit,) = await collection_hits(again.name, "portable")
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
    assert {h.collection for h in await collection_hits("second", "shared")} == {"second"}


async def test_session_search_embeds_once_and_checks_the_model_once(dbos, monkeypatch) -> None:
    """Three collections, one embedding: the query used to be embedded (and the model checked)
    once per collection."""
    from haskie.collection.index import Row

    user = await save_user_settings(UserSettings(embedding="compact"))
    compact = await catalogue.embedding_model(user)
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


async def test_session_search_folds_a_near_duplicate_by_its_vector(
    dbos, models_ready: None, monkeypatch
) -> None:
    """The fold runs on the vectors the index returned: two documents whose chunks embed alike
    come back as one hit that names the other, though their words have little in common. If the
    rows lost their vectors, the search would fall back to words and keep both."""
    from haskie.collection.index import Row

    user = await save_user_settings(UserSettings(embedding="compact"))
    compact = await catalogue.embedding_model(user)
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

    monkeypatch.setattr(retrieval, "embed_query", lambda model, text: vector)

    (hit,) = await flow.chunks(await session.collections_for("s"), "idempotent retries", limit=10)

    assert [ref.document for ref in hit.also_in] == [({"a.md", "b.md"} - {hit.document}).pop()], (
        "the other document, folded in"
    )
    assert len(hit.also_in) == 1
    (folded,) = hit.also_in
    assert folded.to_parent.embedding is not None
    assert folded.to_parent.embedding.alike == pytest.approx(1.0), "by its vector"
    assert folded.to_parent.words.alike < 0.5, "the words differ"


async def test_session_search_reranks_once_over_the_merge(
    dbos, models_ready: None, tmp_path: Path, monkeypatch
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

    def fake_rerank(model: str, accelerator: str, query: str, texts: list[str]) -> list[float]:
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
    sigmoid = [1 / (1 + math.exp(-logit)) for logit in (3, 2)]
    assert [h.score for h in hits] == pytest.approx(sigmoid), (
        "the hit carries the sigmoid of the cross-encoder's logit"
    )


@pytest.mark.parametrize(
    ("name", "with_context", "read"),
    [
        (
            "off, the default: the reranker reads each question alone",
            False,
            {"Why retry?", "How long to wait?"},
        ),
        (
            "on: the context in front of each",
            True,
            {"payments\n\nWhy retry?", "payments\n\nHow long to wait?"},
        ),
    ],
)
async def test_the_reranker_reads_the_shared_context_only_when_set_to(
    dbos,
    models_ready: None,
    tmp_path: Path,
    monkeypatch,
    name: str,
    with_context: bool,
    read: set[str],
) -> None:
    """A cross-encoder matches words: a context every document shares ("ddd" over a DDD book)
    would outrank what each question asks, so by default the reranker, the growth of short
    passages included, reads the question alone."""
    settings = SearchSettings(reranker=Reranker.CROSS_ENCODER, rerank_with_context=with_context)
    await save_user_settings(UserSettings(search=settings))
    await Collection.create("pay")
    body = "# Retries\n\nWhy retry a payment? Wait longer each time.\n\n# Other\n\nx\n"
    doc = await import_document(dbos, "pay.md", body, tmp_path)
    await attach_document(dbos, "pay", doc.name)
    asked: set[str] = set()

    def fake_rerank(model: str, accelerator: str, query: str, texts: list[str]) -> list[float]:
        asked.add(query)
        return [0.0] * len(texts)

    monkeypatch.setattr(embed, "rerank_scores", fake_rerank)

    await flow.answers(["pay"], aspects.questions(["Why retry?", "How long to wait?"], "payments"))

    assert asked == read, name


def _judge(topics: dict[str, str]) -> Any:
    """A reranker that answers logit 5 where the chunk holds the topic word of the question
    (`topics`, question word to chunk word), else -8: sigmoid 0.993 and 0.0003."""

    def rerank(model: str, accelerator: str, query: str, texts: list[str]) -> list[float]:
        words = [word for asked, word in topics.items() if asked in query.lower()]
        return [5.0 if any(word in text.lower() for word in words) else -8.0 for text in texts]

    return rerank


async def test_with_a_reranker_a_question_tags_only_what_it_judged_an_answer(
    dbos, models_ready: None, tmp_path: Path, monkeypatch
) -> None:
    """The reranker scores every chunk against each question: under the floor a chunk is dropped,
    so a question tags only the excerpts it judged an answer, each excerpt scores its best
    question, and a question nothing clears is unanswered rather than given the least bad."""
    await save_user_settings(UserSettings(search=SearchSettings(reranker=Reranker.CROSS_ENCODER)))
    await Collection.create("pay")
    body = (
        "# Retries\n\nRetry a failed payment with backoff.\n\n"
        "# Refunds\n\nRefund a payment to the card it came from.\n\n"
        "# Weather\n\nIt rains on the payment office.\n"
    )
    doc = await import_document(dbos, "pay.md", body, tmp_path)
    await attach_document(dbos, "pay", doc.name)

    monkeypatch.setattr(embed, "rerank_scores", _judge({"retry": "retry", "refund": "refund"}))
    asked = ["Why retry a payment?", "How does a refund reach the card?", "Is there a tax?"]

    answer = await flow.answers(["pay"], aspects.questions(asked, None))

    found = {one.header: (one.aspects, one.aspect_scores, one.score) for one in answer.excerpts}
    top = 1 / (1 + math.exp(-5))
    assert found == {
        "Retries": ([asked[0]], {asked[0]: pytest.approx(top)}, pytest.approx(top)),
        "Refunds": ([asked[1]], {asked[1]: pytest.approx(top)}, pytest.approx(top)),
    }, "no weather: every question judged it no answer"
    assert answer.uncovered == [asked[2]], "nothing about a tax, so it is not answered"


async def test_an_excerpts_search_stems_its_answer_once_on_the_event_loop(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """An excerpts search reads the kept sections' words for the probe, then the answer's for its
    report. `probe.stem` remembers each word, so the second read stems nothing again and both run
    on the event loop, with no wait for a slot of the CPU budget that indexing may hold."""
    import threading

    from haskie.search import probe

    await Collection.create("pay")
    body = "# Retries\n\nRetry a failed payment with backoff.\n\n# Refunds\n\nRefund a payment.\n"
    doc = await import_document(dbos, "pay.md", body, tmp_path)
    await attach_document(dbos, "pay", doc.name)
    probe.stem.cache_clear()
    loop_thread = threading.get_ident()
    reads: list[tuple[int, int]] = []  # (thread, words stemmed afresh)
    real = probe.vocabulary

    def recorded(texts):
        before = probe.stem.cache_info().misses
        found = real(texts)
        reads.append((threading.get_ident(), probe.stem.cache_info().misses - before))
        return found

    monkeypatch.setattr(probe, "vocabulary", recorded)

    answer = await flow.answers(["pay"], aspects.questions(["Why retry a payment?"], None))

    assert answer.excerpts, "the search found the section it stems"
    (probe_on, probed), (report_on, reported) = reads  # once for the probe, once for the report
    assert probed > 0 and reported == 0, "the report's words were stemmed for the probe"
    assert probe_on == report_on == loop_thread, "both on the event loop"


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

    async def boom(self, query, vector, settings_, limit, vectors=True):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(CollectionIndex, "search_rows", boom)

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="index unreadable"):
            await flow.chunks(await session.collections_for("s"), "shared")

    assert "session_collection_search_failed" in events(caplog)
