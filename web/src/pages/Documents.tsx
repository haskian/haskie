import { ExternalLink, Library, Plus, Trash2, Upload, X } from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  api,
  type DocumentStatus,
  type Document,
  type EmbeddingEntry,
  type Staged,
} from "../api";
import type { PageProps } from "../App";
import { errorText, bytes, dateTime, day, matchesText, needleOf } from "../format";
import { useOperation } from "../hooks/useOperation";
import { useOptions } from "../hooks/useOptions";
import { usePaged } from "../hooks/usePaged";
import { usePoll } from "../hooks/usePoll";
import { useRun } from "../hooks/useRun";
import { href, navigate, type Route } from "../router";
import {
  documentIcon,
  DescriptionBox,
  DocumentPanes,
  DropOverlay,
  GallerySection,
  groupByRange,
  Kv,
  Modal,
  Shell,
  SearchBox,
  Tabs,
  Tile,
  type TabDef,
} from "../ui";
import "./Documents.css";
import { embeddingLabel } from "./documents/embedding";
import { groupByDay, groupByStatus } from "./documents/group";

type GroupBy = "status" | "name" | "day";
const GROUPS: { id: GroupBy; label: string }[] = [
  { id: "status", label: "Status" },
  { id: "name", label: "Name" },
  { id: "day", label: "Date" },
];
type StagedFile = Staged & { name: string }; // the name the document will get; the filename until edited

/** Every document in the home, imported once and shared by the collections that hold it. */
export function Documents({
  route,
  counts,
}: PageProps<Extract<Route, { name: "documents" }>>) {
  const [groupBy, setGroupBy] = useState<GroupBy>("status");
  const [search, setSearch] = useState("");
  const [path, setPath] = useState("");
  const [adding, setAdding] = useState(false);
  const [staged, setStaged] = useState<StagedFile[]>([]); // uploaded, named, not yet imported
  const fileInput = useRef<HTMLInputElement>(null);

  const options = useOptions();
  const docs = usePaged(api.documents, { sort: "name", pageSize: 500 });
  const refresh = docs.refresh;
  const { run, busy, error } = useRun(refresh);

  // Anything still in the pipeline keeps the listing fresh; so does a deletion until it is gone.
  const active = docs.items.some(
    (doc) =>
      options.active_document_statuses.includes(doc.status) ||
      doc.status === "deleting",
  );
  usePoll(active, refresh);

  // Upload is two steps per file: the bytes are staged, then named and imported. Staging happens
  // on drop, one file at a time so a batch of large files does not open a request each; the
  // import waits for the names to be confirmed in the modal.
  const upload = useCallback(
    (files: File[]) =>
      run(async () => {
        setAdding(true);
        for (const file of files) {
          const one = await api.stageUpload(file);
          setStaged((rows) => [...rows, { ...one, name: one.filename }]);
        }
      }),
    [run],
  );

  const importStaged = () =>
    run(async () => {
      await Promise.all(
        staged.map((one) =>
          api.importStaged({ staging_id: one.staging_id, name: one.name.trim() }),
        ),
      );
      setStaged([]);
      setAdding(false);
    });

  const rename = (id: string, name: string) =>
    setStaged((rows) =>
      rows.map((one) => (one.staging_id === id ? { ...one, name } : one)),
    );
  // ponytail: dropping a row only forgets it here; the staged bytes age out on the server.
  const forget = (id: string) =>
    setStaged((rows) => rows.filter((one) => one.staging_id !== id));
  const namesOk =
    staged.length > 0 && staged.every((one) => one.name.trim() !== "");

  const importPath = (event: FormEvent) => {
    event.preventDefault();
    void run(async () => {
      await api.importPath(path.trim());
      setPath("");
      setAdding(false);
    });
  };

  const closeModal = useCallback(() => navigate({ name: "documents" }), []);

  // The filter is client-side over the rows already loaded: the backend has no name search, and a
  // page of 500 is what the gallery shows anyway.
  const needle = needleOf(search);
  const groups = useMemo(() => {
    const visible = docs.items.filter((doc) =>
      matchesText(needle, doc.name, doc.description),
    );
    if (groupBy === "status") return groupByStatus(visible, options.document_statuses);
    if (groupBy === "day") return groupByDay(visible);
    return groupByRange(visible, (doc) => doc.name);
  }, [docs.items, needle, groupBy, options]);

  const side = (
      <section>
        <span className="label label-mono">Group by</span>
        <nav className="nav" id="group-by">
          {GROUPS.map((group) => (
            <button
              key={group.id}
              className="nav-item"
              type="button"
              data-group={group.id}
              aria-current={groupBy === group.id ? "true" : undefined}
              onClick={() => setGroupBy(group.id)}
            >
              {group.label}
            </button>
          ))}
        </nav>
      </section>
  );

  return (
    <Shell current={route.name} counts={counts} side={side}>
      <div className="gallery-sections sections">
        <SearchBox id="search" value={search} onChange={setSearch} placeholder="Search documents" />
        {error !== null && <p className="muted">{error}</p>}
        {docs.error !== null && <p className="muted">{docs.error}</p>}
        <GallerySection label="New" large>
          <Tile
            icon={Plus}
            name="Add documents"
            sub="Upload or import"
            hint="Drop files anywhere, upload, or import a path."
            add
            onClick={() => setAdding(true)}
          />
        </GallerySection>
        {groups.map((group) => (
          <GallerySection key={group.label} label={group.label} large>
            {group.items.map((doc) => (
              <Tile
                key={doc.name}
                icon={documentIcon(doc.suffix)}
                name={doc.name}
                sub={
                  <>
                    <Library className="glyph" /> {doc.collections}
                    {doc.status !== "imported" && ` · ${doc.status}`}
                  </>
                }
                meta={day(doc.created_at)}
                hint={doc.description || "No description"}
                onClick={() => navigate({ name: "documents", document: doc.name })}
              />
            ))}
          </GallerySection>
        ))}
        {docs.hasMore && (
          <button
            className="btn btn-ghost"
            type="button"
            disabled={docs.loading}
            onClick={docs.loadMore}
          >
            Load more
          </button>
        )}
      </div>
      <DropOverlay onFiles={upload} />
      <Modal
        open={adding}
        onClose={() => setAdding(false)}
        title="Add documents"
        subtitle="import"
      >
        <div className="add-documents">
          <input
            type="file"
            multiple
            hidden
            ref={fileInput}
            onChange={(event) => {
              const files = [...(event.target.files ?? [])];
              if (files.length > 0) void upload(files);
              event.target.value = ""; // the same file twice in a row is still a change
            }}
          />
          <div className="field">
            <span className="label">Files</span>
            <div className="row">
              <button
                className="btn"
                type="button"
                disabled={busy}
                onClick={() => fileInput.current?.click()}
              >
                <Upload className="icon" />
                Choose files
              </button>
              <span className="muted">or drop them anywhere on the page</span>
            </div>
          </div>
          {staged.length > 0 && (
            <div className="field">
              <span className="label">Ready to import · {staged.length}</span>
              {/* Nothing is converted or embedded until the names below are confirmed. */}
              <ul className="list staged">
                {staged.map((one) => (
                  <li key={one.staging_id} className="list-item">
                    <input
                      className="input"
                      value={one.name}
                      aria-label="Document name"
                      onChange={(event) =>
                        rename(one.staging_id, event.target.value)
                      }
                    />
                    <span className="mono muted">{bytes.format(one.size)}</span>
                    <button
                      className="btn btn-ghost"
                      type="button"
                      aria-label="Remove"
                      onClick={() => forget(one.staging_id)}
                    >
                      <X className="icon" />
                    </button>
                  </li>
                ))}
              </ul>
              <div className="row">
                <button
                  className="btn btn-primary"
                  type="button"
                  disabled={busy || !namesOk}
                  onClick={importStaged}
                >
                  Import {staged.length}{" "}
                  {staged.length === 1 ? "document" : "documents"}
                </button>
              </div>
            </div>
          )}
          {/* A form, so Enter imports the way the browser already does it. */}
          <form className="field" onSubmit={importPath}>
            <span className="label">Path</span>
            <input
              className="input"
              placeholder="File or folder the server can read"
              value={path}
              onChange={(event) => setPath(event.target.value)}
            />
            <div className="row">
              <button
                className="btn"
                type="submit"
                disabled={busy || path.trim() === ""}
              >
                Import path
              </button>
            </div>
          </form>
          {error !== null && <p className="muted">{error}</p>}
        </div>
      </Modal>
      {/* Keyed by name: another document starts its panels and its reads over. */}
      <DocumentModal
        key={route.document}
        doc={route.document ?? null}
        onClose={closeModal}
        onChanged={refresh}
      />
    </Shell>
  );
}

// An import can only be re-run from a state it stopped in; the backend refuses every other status.
const RETRYABLE: readonly DocumentStatus[] = ["error", "cancelled"];

const CONTENT_TAB = "modal-content";
const COLLECTIONS_TAB = "modal-collections";
const IMPORT_TAB = "modal-import";
const MODAL_TABS: TabDef[] = [
  { id: CONTENT_TAB, label: "Content" },
  { id: COLLECTIONS_TAB, label: "Collections" },
  { id: IMPORT_TAB, label: "Info" },
];

/** One document, opened from the gallery: what it says, and how it was imported. */
function DocumentModal({
  doc,
  onClose,
  onChanged,
}: {
  doc: string | null;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [tab, setTab] = useState(CONTENT_TAB);
  const [row, setRow] = useState<Document | null>(null);
  const [collections, setCollections] = useState<string[]>([]);
  const [embeddings, setEmbeddings] = useState<EmbeddingEntry[]>([]);

  // The three reads the modal needs, in one round: the row itself, who holds it, what is cached.
  const load = useCallback((): Promise<void> => {
    if (doc === null) return Promise.resolve();
    return Promise.all([
      api.document(doc),
      api.documentCollections(doc),
      api.documentEmbeddings(doc),
    ]).then(([fetched, names, cached]) => {
      setRow(fetched);
      setCollections(names);
      setEmbeddings(cached);
    });
  }, [doc]);

  const { run, busy, error, setError } = useRun(load);

  useEffect(() => {
    load().catch((cause: unknown) => setError(errorText(cause)));
  }, [load, setError]);

  // A deletion is accepted (202) and runs in the background: the modal follows it, then
  // closes and lets the listing re-read itself.
  const onDeleted = useCallback(() => {
    onClose();
    onChanged();
  }, [onClose, onChanged]);
  const deletion = useOperation(onDeleted, setError);

  const remove = () => {
    if (doc === null) return;
    if (!window.confirm(`Delete "${doc}" and remove it from every collection?`))
      return;
    deletion
      .start(() => api.deleteDocument(doc))
      .catch((cause: unknown) => setError(errorText(cause)));
  };

  const rows: [string, ReactNode][] =
    row === null
      ? []
      : [
          ["Imported", dateTime(row.created_at)],
          ["Updated", dateTime(row.updated_at)],
          [
            "Status",
            <>
              {row.status}
              {row.error !== null && (
                <>
                  {" "}
                  · <span className="code">{row.error}</span>
                </>
              )}
            </>,
          ],
          ["Source", `${row.suffix} · ${bytes.format(row.size)}`],
          ["Parser", row.parser],
          ["Skip OCR pages", row.skip_ocr_pages ? "yes" : "no"],
          [
            "Embeddings",
            embeddings.length === 0
              ? "none cached"
              : embeddings.map((entry) => (
                  <div key={entry.id}>{embeddingLabel(entry)}</div>
                )),
          ],
        ];

  return (
    <Modal
      open={doc !== null}
      onClose={onClose}
      title={doc ?? ""}
      subtitle={
        row === null ? undefined : `${bytes.format(row.size)} · ${row.suffix}`
      }
    >
      <Tabs tabs={MODAL_TABS} selected={tab} onSelect={setTab} />
      <div
        id={CONTENT_TAB}
        role="tabpanel"
        className="modal-panel"
        hidden={tab !== CONTENT_TAB}
      >
        {row !== null && <DocumentPanes doc={row.name} preview={row.preview} />}
      </div>
      <div
        id={COLLECTIONS_TAB}
        role="tabpanel"
        hidden={tab !== COLLECTIONS_TAB}
      >
        {collections.length === 0 ? (
          <p className="muted">In no collection yet.</p>
        ) : (
          <ul className="list">
            {collections.map((name) => (
              <li className="list-item" key={name}>
                <Library className="icon" />
                <span className="list-text">
                  <a href={href({ name: "collections", collection: name })}>
                    {name}
                  </a>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div id={IMPORT_TAB} role="tabpanel" hidden={tab !== IMPORT_TAB}>
        <div className="split">
          <section className="pane">
            <span className="pane-head mono muted">Details</span>
            <div className="pane-body">
              <Kv rows={rows} />
            </div>
          </section>
          <section className="pane">
            <span className="pane-head mono muted">Description</span>
            <div className="pane-body">
              {row !== null && (
                <DescriptionBox
                  value={row.description}
                  placeholder="What this document is about"
                  onSave={(next) =>
                    void run(() => api.describeDocument(row.name, next))
                  }
                />
              )}
            </div>
          </section>
        </div>
        {row !== null && (
          <div className="row row-loose">
            <a
              className="btn"
              href={api.sourceUrl(row.name)}
              target="_blank"
              rel="noreferrer"
            >
              <ExternalLink className="icon" />
              Open original
            </a>
            {RETRYABLE.includes(row.status) && (
              <button
                className="btn"
                type="button"
                disabled={busy}
                onClick={() => run(() => api.reimportDocument(row.name))}
              >
                Retry import
              </button>
            )}
            <button
              className="btn btn-ghost"
              type="button"
              disabled={busy || deletion.running}
              onClick={remove}
            >
              <Trash2 className="icon" />
              Delete
            </button>
            {deletion.running && <span className="muted">deleting…</span>}
          </div>
        )}
        {error !== null && <p className="muted">{error}</p>}
      </div>
    </Modal>
  );
}
