# -*- coding: utf-8 -*-
"""Regenerate _probe/ui_app.html from the CURRENT index.html.

Run from the repo root:  python _probe/make_ui_probe.py

ui_app.html is what every UI probe (btn_check / ui_check / ui_measure /
ui_shot) actually loads. It keeps the real stylesheet, real tfjs and an
unmodified /script.js, and swaps in fixed-response stubs for window.cocoSsd and
window.tflite ONLY - so the probes test the production markup and the
production pipeline, on a machine with no camera and no GPU.

Re-run this after ANY change to index.html, or the probes will keep measuring
the previous markup.
"""
import io
import re

app = io.open('index.html', encoding='utf-8').read()
old = io.open('_probe/ui_app.html', encoding='utf-8').read()

# The stub block is the only hand-written part; carry it across verbatim.
m = re.search(r'\n    <!-- UI PROBE STUBS\..*?\n    </script>\n', old, re.S)
assert m, 'stub block not found in existing _probe/ui_app.html'
stubs = m.group(0)

out = app.replace('href="style.css"', 'href="/style.css"')
out = out.replace('src="script.js"', 'src="/script.js"')
anchor = '\n    <!-- Application Script -->'
assert out.count(anchor) == 1, 'application script anchor not found in index.html'
out = out.replace(anchor, stubs + anchor)

io.open('_probe/ui_app.html', 'w', encoding='utf-8', newline='').write(out)
print('_probe/ui_app.html regenerated from index.html')
