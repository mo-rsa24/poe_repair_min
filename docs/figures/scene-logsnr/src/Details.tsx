import type { ReactNode } from 'react'

/**
 * Everything true that is not first-read material: caveats, provenance, what
 * a panel may not claim. Closed by default, one click away, never dropped.
 */
export function Details({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="disclose">
      <summary>{summary}</summary>
      <div>{children}</div>
    </details>
  )
}
