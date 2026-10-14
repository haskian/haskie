export interface TabDef {
  id: string // also the id of the panel the caller renders
  label: string
  count?: number // shown after the label as a tag, the way a question shows Q1
}

/**
 * The tab strip only. Panels stay with the caller, as
 * `<div id={id} role="tabpanel" hidden={selected !== id}>`, so a page decides what a panel costs.
 */
export function Tabs({
  tabs,
  selected,
  onSelect,
}: {
  tabs: TabDef[]
  selected: string
  onSelect: (id: string) => void
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className="tab"
          type="button"
          role="tab"
          aria-selected={tab.id === selected}
          aria-controls={tab.id}
          onClick={() => onSelect(tab.id)}
        >
          {tab.label}
          {/* The space keeps the label and the count two words to a screen reader. */}
          {tab.count !== undefined && <> <span className="descriptor">{tab.count}</span></>}
        </button>
      ))}
    </div>
  )
}
