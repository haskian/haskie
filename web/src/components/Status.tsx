// The "all statuses" dropdown above a listing. `counts` is shown per status where the backend
// counts them; the empty value is "no filter", which is not the same request as any status.
export function StatusFilter<S extends string>({
  value,
  statuses,
  counts,
  onChange,
}: {
  value: S | ''
  statuses: readonly S[]
  counts?: Record<string, number>
  onChange: (status: S | '') => void
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value as S | '')}>
      <option value="">all statuses</option>
      {statuses.map((s) => (
        <option key={s} value={s}>
          {s}{counts && ` (${counts[s] ?? 0})`}
        </option>
      ))}
    </select>
  )
}

// The status cell of a listing row: the status, and the failure under it when there is one.
export function StatusCell({ status, error }: { status: string; error: string | null }) {
  return (
    <td className={status === 'error' ? 'error' : 'muted'}>
      {status}
      {error && <pre className="error-detail">{error}</pre>}
    </td>
  )
}
