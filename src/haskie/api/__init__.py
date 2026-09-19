"""The HTTP/MCP surface, one module per feature.

`app.py` owns the application: the request context, the error mapping and `create_app`. Each
module here owns one feature's routes and the request/response shapes only that feature uses;
what two of them share lives in `common.py`.
"""

from litestar.types import ControllerRouterHandler

from haskie.api import collections, documents, jobs, search, settings

# Order is the order the routes appear in the OpenAPI document, so it follows the UI: set up,
# then the documents, then the collections holding them, then the work, then searching.
ROUTE_HANDLERS: list[ControllerRouterHandler] = [
    settings.get_status,
    settings.post_init,
    settings.get_settings,
    settings.put_settings,
    settings.get_options,
    documents.stage_document,
    documents.import_document,
    documents.list_documents,
    documents.get_document,
    documents.delete_document,
    documents.reimport_document,
    documents.list_document_collections,
    documents.list_document_embeddings,
    documents.get_source,
    documents.get_preview,
    documents.get_markdown,
    documents.describe_document,
    collections.list_collections,
    collections.create_collection,
    collections.get_collection,
    collections.delete_collection,
    collections.search_collection,
    collections.put_collection_settings,
    collections.describe_collection,
    collections.index_collection,
    collections.list_collection_documents,
    collections.add_document,
    collections.remove_document,
    collections.index_collection_document,
    jobs.list_jobs,
    jobs.list_jobs_by_kind,
    jobs.list_job_kinds,
    jobs.get_activity,
    jobs.list_job_tasks,
    jobs.get_job_progress,
    jobs.delete_job,
    search.list_sessions,
    search.put_session,
    search.search_session,
    search.search_text,
    search.search_documents,
]
