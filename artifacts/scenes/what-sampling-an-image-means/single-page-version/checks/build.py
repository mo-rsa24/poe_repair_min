# Build the published page from the authored source: inline KaTeX's stylesheet and the two photos.
# Run from anywhere: python3 checks/build.py
import base64, pathlib
here = pathlib.Path(__file__).resolve().parent.parent
src = (here / 'what-sampling-an-image-means.src.html').read_text()
kat = (pathlib.Path.home() / '.claude/skills/picture-speak/assets/katex-inline-style.html').read_text()
b64 = lambda p: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
out = (src.replace('%%KATEX_CSS%%', kat)
          .replace('%%DOG_B64%%', b64(here / 'assets/labrador-128px.png'))
          .replace('%%DOGS_B64%%', b64(here / 'assets/four-training-dogs-48px.png')))
(here / 'what-sampling-an-image-means.html').write_text(out)
print(len(out), 'bytes')
