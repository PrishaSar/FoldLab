"""Self-contained py3Dmol page colored by the H/P class.

The 3Dmol script is cached under data/ and inlined, so the HTML file opens
without fetching the viewer again.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

from foldlab.structure import DATA_DIR

JS_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
JS_CACHE = DATA_DIR / "3dmol-min.js"
H_COLOR = "#c2410c"
P_COLOR = "#0369a1"


def cached_viewer_js() -> str:
    if not JS_CACHE.is_file() or JS_CACHE.stat().st_size == 0:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(JS_URL, JS_CACHE)
    return JS_CACHE.read_text()


def write_hp_view(
    path: Path,
    pdb_text: str,
    chain_id: str,
    classes: list[tuple[int, str]],
    title: str,
) -> Path:
    """Write one HTML file. ``classes`` is (residue number, 'H' or 'P')."""
    import py3Dmol

    view = py3Dmol.view(width=960, height=680, js=JS_URL)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "#d6d3d1"}})
    for resseq, klass in classes:
        color = H_COLOR if klass == "H" else P_COLOR
        view.setStyle({"chain": chain_id, "resi": int(resseq)}, {"cartoon": {"color": color}})
    view.zoomTo()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view.write_html(str(path))
    html = path.read_text()
    caption = (
        '<div style="font-family: Georgia, serif; background:#fffbf5; color:#1c1917;'
        ' padding:14px 18px 8px;">'
        f"<strong>{title}</strong>"
        '<span style="color:#57534e;"> · rust H (Kyte–Doolittle &gt; 0) · blue P</span>'
        "</div>\n"
    )
    library = "<script>\n" + cached_viewer_js() + "\n</script>\n<script>var $3Dmolpromise = Promise.resolve();</script>\n"
    html = html.replace(
        '<body style="margin: 0; padding: 0; display: block;">',
        '<body style="margin: 0; padding: 0; background:#fffbf5;">\n' + caption + library,
        1,
    )
    path.write_text(html)
    return path
