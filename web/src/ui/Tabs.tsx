export interface TabDef {
  id: string // also the id of the panel the caller renders
  label: string
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
        </button>
      ))}
    </div>
  )
}
