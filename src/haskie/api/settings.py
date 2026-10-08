"""Init, user settings and the option catalogue the UI renders its forms from."""

import msgspec
from litestar import get, post, put
from litestar.datastructures import State

from haskie import audit, home
from haskie.catalogue import catalogue
from haskie.catalogue.catalogue import EmbedderMetadata, EmbeddingModel, RerankerMetadata
from haskie.document.document import ACTIVE_DOCUMENT_STATUSES, DOCUMENT_STATUSES, DocumentStatus
from haskie.errors import Conflict
from haskie.indexing import gguf_models, hardware, models, workflows
from haskie.indexing.dbos_names import ACTIVE_STATUS, RunStatus
from haskie.settings import (
    NO_EMBEDDING,
    Accelerator,
    Chunker,
    Describer,
    Descriptors,
    FieldDoc,
    FillValues,
    Fusion,
    Parser,
    PipelineSettings,
    Reranker,
    ScoreFold,
    SearchMode,
    SearchSettings,
    UserSettings,
    docs,
    first_run_search,
    init_user_settings,
    load_user_settings,
    load_user_settings_or_none,
    save_user_settings,
    settings_problem,
)

# The app's state key saying whether it serves the web UI: `create_app` decides, the status says.
WEB_UI_STATE = "web_ui"


class Status(msgspec.Struct):
    initialized: bool
    home: str
    embedding: EmbeddingModel | None
    models: list[models.ModelStatus]  # download/load state of every model the settings need
    settings_error: str | None = None  # stored settings unreadable; defaults are in use
    web_ui: bool = False  # whether this server serves the built web UI at `/`


class Init(msgspec.Struct):
    """The first run's choices: what to embed with, how to search, and who writes the section
    descriptors. Everything else starts at its default and is changed in the settings later."""

    profile: str  # a key of `Options.embedding_profiles`
    search: SearchSettings = msgspec.field(default_factory=first_run_search)
    descriptors: Descriptors = PipelineSettings().descriptors  # one of `Options.descriptors`
    describer: Describer = PipelineSettings().describer  # one of `Options.describers`


class Options(msgspec.Struct):
    """Every choice the UI offers, from the catalogue and the constants the backend validates
    against, so nothing is spelled a second time in the frontend."""

    parsers: tuple[Parser, ...]
    chunkers: tuple[Chunker, ...]
    accelerators: tuple[Accelerator, ...]
    descriptors: tuple[Descriptors, ...]  # llm only where a describer runs (`hardware.device`)
    describers: tuple[Describer, ...]  # the llm strategy's models that run here
    search_modes: tuple[SearchMode, ...]
    fusions: tuple[Fusion, ...]
    score_folds: tuple[ScoreFold, ...]
    fill_values: tuple[FillValues, ...]
    rerankers: tuple[Reranker, ...]
    reranker_models: tuple[str, ...]
    reranker_metadata: dict[str, RerankerMetadata]  # every reranker model, offered here or not
    docs: dict[str, FieldDoc]  # title + definition per setting key, e.g. "conversion.chunk_size"
    # the profiles offered here, in picker order: "none" first, then those that can load here
    embedding_profiles: dict[str, EmbeddingModel | None]
    embedding_metadata: dict[str, EmbedderMetadata]  # every profile's, offered here or not
    document_statuses: tuple[DocumentStatus, ...]
    # in the import pipeline: a poll waits on them
    active_document_statuses: tuple[DocumentStatus, ...]
    active_run_statuses: tuple[RunStatus, ...]  # run statuses that are still on their way


def _changed_fields(before: msgspec.Struct, after: msgspec.Struct, prefix: str = "") -> list[str]:
    """Dotted names of the settings that differ, for the audit trail. Names only: a value may
    be a path or a model id, which does not belong in an audit record."""
    changed: list[str] = []
    for field in msgspec.structs.fields(after):
        old, new = getattr(before, field.name), getattr(after, field.name)
        if old == new:
            continue
        if isinstance(old, msgspec.Struct) and isinstance(new, msgspec.Struct):
            changed.extend(_changed_fields(old, new, f"{prefix}{field.name}."))
        else:
            changed.append(f"{prefix}{field.name}")
    return changed


@get("/api/status")
async def get_status(state: State) -> Status:
    saved = await load_user_settings_or_none()
    return Status(
        initialized=saved is not None,
        home=str(home.HOME),
        embedding=(await catalogue.embedding_model(saved)) if saved else None,
        models=(await models.model_statuses()) if saved else [],
        settings_error=settings_problem(),
        web_ui=state.get(WEB_UI_STATE, False),
    )


@post("/api/init")
@audit.audited("settings.init")
async def post_init(data: Init) -> UserSettings:
    """First run: pick the embedding profile, the search mode, the reranker and the descriptors.
    The models download in the background; poll /api/status -> models."""
    audit.attach(profile=data.profile)
    settings = UserSettings(
        embedding=data.profile,
        search=data.search,
        pipeline=PipelineSettings(descriptors=data.descriptors, describer=data.describer),
    )
    await catalogue.check(settings)
    if not await init_user_settings(settings):
        raise Conflict("already initialized; change embedding via settings and reindex")
    await workflows.apply_settings(settings)
    return settings


@get("/api/settings")
async def get_settings() -> UserSettings:
    """The stored settings; before the first run, the ones it starts from."""
    return await load_user_settings_or_none() or UserSettings(search=first_run_search())


@put("/api/settings")
@audit.audited("settings.update")
async def put_settings(data: UserSettings) -> UserSettings:
    """Saves, then applies: every queue limit follows its setting, new models start downloading.
    Values out of range are rejected while the body is decoded, and models the catalogue does not
    hold before the save, so nothing invalid is stored."""
    await catalogue.check(data)
    changed = _changed_fields(await load_user_settings(), data)
    if changed:
        audit.attach(changed=",".join(changed))
    settings = await save_user_settings(data)
    await workflows.apply_settings(settings)
    return settings


@get("/api/options")
async def get_options() -> Options:
    embedders = await catalogue.embedders()
    rerankers = await catalogue.rerankers()
    accelerator = (await load_user_settings()).pipeline.accelerator
    describers = tuple(
        one
        for one, generator in gguf_models.DESCRIBERS.items()
        if hardware.device(generator.name, accelerator) is not None
    )
    return Options(
        parsers=tuple(Parser),
        chunkers=tuple(Chunker),
        accelerators=tuple(Accelerator),
        descriptors=tuple(one for one in Descriptors if one != Descriptors.LLM or describers),
        describers=describers,
        search_modes=tuple(SearchMode),
        fusions=tuple(Fusion),
        score_folds=tuple(ScoreFold),
        fill_values=tuple(FillValues),
        rerankers=tuple(Reranker),
        # a model is offered only where it runs under the hardware setting; one already chosen
        # still validates, and fails to load with the reason (`hardware.device`)
        reranker_models=tuple(
            name for name in rerankers if hardware.device(name, accelerator) is not None
        ),
        reranker_metadata=rerankers,
        docs=docs(),
        embedding_profiles={
            NO_EMBEDDING: None,
            **{
                p: model
                for p, model in embedders.items()
                if hardware.device(model.name, accelerator) is not None
            },
        },
        embedding_metadata=await catalogue.embedding_metadata(),
        document_statuses=DOCUMENT_STATUSES,
        active_document_statuses=ACTIVE_DOCUMENT_STATUSES,
        active_run_statuses=tuple(RunStatus(status) for status in ACTIVE_STATUS),
    )
