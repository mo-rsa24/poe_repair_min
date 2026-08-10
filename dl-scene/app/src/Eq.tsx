import katex from "katex";
import "katex/dist/katex.min.css";

export function Eq({ tex, block }: { tex: string; block?: boolean }) {
  const html = katex.renderToString(tex, {
    throwOnError: false,
    displayMode: !!block,
  });
  return (
    <span
      className={block ? "eq" : undefined}
      style={block ? { display: "block" } : undefined}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
