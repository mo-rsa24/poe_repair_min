# Build the three pages of "From 173 cm to a Dog" from the shared shell, the shared library and each scene's frames.
import base64, pathlib, sys
src = pathlib.Path(__file__).resolve().parent
here = src.parent  # the scene folder: built pages land here, photos sit in assets/
kat = (pathlib.Path.home() / '.claude/skills/picture-speak/assets/katex-inline-style.html').read_text()
b64 = lambda p: 'data:image/png;base64,' + base64.b64encode((here / p).read_bytes()).decode()
shell, lib = (src / 'shell.html').read_text(), (src / 'lib.js').read_text()
CRED = ('<p>Photos, all from Wikimedia Commons: "YellowLabradorLooking new.jpg" by Djmirko (derivative work), CC BY-SA 3.0; '
        '"Golde33443.jpg" by Golden Trvs Gol twister, CC BY-SA 3.0; "Border Collie liver portrait.jpg" by John Haslam, CC BY 2.0; '
        '"Siberian-husky.jpg" by Utopialand, CC BY-SA 4.0. Each is cropped square and resized, to 128 × 128 for the Labrador and 48 × 48 for the four training photos.</p>')
PAGES = [
  (1, 'index.html', 'From 173 cm to a Dog', 'One number: from your height to a sample', 'Step 4’s noise schedule is not used here', ''),
  (2, 'many-numbers.html', 'Many Numbers', 'Many numbers: an image is a point', 'a 2 × 2 toy with two real images chosen by hand, the Labrador photo, and Step 4’s cosine schedule', CRED),
  (3, 'going-back.html', 'Going Back', 'Going back: from noise to a sample', 'the four dog photos and Step 4’s cosine schedule (T = 100, offset 0.008, β clipped at 0.999)', CRED),
]
only = set(sys.argv[1:])
for no, name, title, h1, sources, cred in PAGES:
    js = src / f'scene{no}.js'
    if (only and str(no) not in only) or not js.exists(): continue
    scene = js.read_text().replace('%%DOG_B64%%', b64('assets/labrador-128px.png')).replace('%%DOGS_B64%%', b64('assets/four-training-dogs-48px.png'))
    out = (shell.replace('%%TITLE%%', title).replace('%%H1%%', h1).replace('%%NO%%', str(no)).replace('%%SOURCES%%', sources)
                .replace('%%CREDITS%%', cred).replace('%%KATEX_CSS%%', kat).replace('%%LIB%%', lib).replace('%%SCENE%%', scene))
    if no != 1:  # a sibling page is served as it is, without the artifact's wrapping skeleton
        out = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n' + out
    (here / name).write_text(out); print(name, len(out))
