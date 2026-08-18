import katex from 'katex'
import 'katex/dist/katex.min.css'

export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = katex.renderToString(children, {
    displayMode: block,
    throwOnError: false,
    trust: true,
  })
  return <span className={block ? 'tex-block' : 'tex'} dangerouslySetInnerHTML={{ __html: html }} />
}
