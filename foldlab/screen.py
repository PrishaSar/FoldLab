"""Local page for the folds, the protein pictures, and the saved score.

Start it from the repository root:

    python -m foldlab.screen

The page is http://127.0.0.1:8765. It calls the same search and figure
code as the command line. Transformer score shows the numbers already
written down. Training runs only from the separate button.
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
HOST = "127.0.0.1"
PORT = 8765

# Numbers from docs/hp-transformer.md. The score button returns these
# and does not train.
SAVED_SCORE = {
    "mae": 0.4984,
    "baseline_mae": 1.4658,
    "accuracy": 0.8679,
    "baseline_accuracy": 0.6981,
    "threshold": -3.0,
    "mean_energy": -3.6872,
    "n": 264,
    "n_train": 211,
    "n_test": 53,
}

FILES = {
    "demo-short.gif": "image/gif",
    "demo-long.gif": "image/gif",
    "lysozyme-contact-map.png": "image/png",
    "lysozyme-hydrophobicity.png": "image/png",
    "lysozyme-hp-core.html": "text/html; charset=utf-8",
    "myoglobin-contact-map.png": "image/png",
    "contact-compare.png": "image/png",
    "sickle-cell-glu6val.png": "image/png",
}

_PROTEIN_PATH = re.compile(
    r"^/proteins/([0-9][a-z0-9]{3}-[a-z0-9])/(contact\.png|oiliness\.png|view\.html)$"
)
_PROTEIN_TYPES = {
    "contact.png": "image/png",
    "oiliness.png": "image/png",
    "view.html": "text/html; charset=utf-8",
}

_JOB = threading.Lock()

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FoldLab</title>
<style>
  :root { color-scheme: light; }
  body {
    margin: 0;
    background: #f3efe6;
    color: #1c1917;
    font: 18px/1.45 Georgia, "Iowan Old Style", serif;
  }
  main { max-width: 42rem; margin: 0 auto; padding: 1.75rem 1.1rem 3rem; }
  h1 { font-size: 2rem; font-weight: normal; margin: 0 0 0.35rem; }
  p { margin: 0.4rem 0; }
  .lead { margin-bottom: 1.1rem; }
  .actions { display: flex; flex-wrap: wrap; gap: 0.45rem; }
  button {
    font: inherit;
    background: #1c1917;
    color: #faf7f2;
    border: 1px solid #1c1917;
    border-radius: 4px;
    padding: 0.45rem 0.75rem;
    cursor: pointer;
  }
  button.quiet { background: transparent; color: #1c1917; }
  button:disabled { opacity: 0.45; cursor: wait; }
  #status { min-height: 1.45em; margin: 0.9rem 0 0.3rem; }
  figure { margin: 0.8rem 0; }
  img, .frame {
    width: 100%;
    height: auto;
    background: white;
    border: 1px solid #d6d0c4;
  }
  iframe.viewer {
    width: 100%;
    height: 720px;
    margin-top: 0.8rem;
    background: white;
    border: 1px solid #d6d0c4;
  }
  form.load {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
    margin: 0 0 0.35rem;
  }
  input[name="code"] {
    font: inherit;
    width: 9rem;
    padding: 0.4rem 0.55rem;
    border: 1px solid #1c1917;
    border-radius: 4px;
    background: white;
  }
  @media (max-width: 700px) {
    iframe.viewer { height: 480px; }
  }
  figcaption { font-size: 0.95rem; margin-top: 0.3rem; }
  a { color: #9a3412; }
  .note { font-size: 0.95rem; color: #44403c; margin-top: 0.8rem; }
</style>
</head>
<body>
<main>
  <h1>FoldLab</h1>
  <p class="lead">Load a protein by its code, or play with the toy folds below.</p>
  <form id="load" class="load">
    <label for="code">Protein code</label>
    <input id="code" name="code" placeholder="1LYZ" maxlength="16" autocomplete="off">
    <button type="submit">Load protein</button>
  </form>
  <p class="note">A 4-character code from the public structure archive. For one chain, type 2HHB B.</p>
  <div class="actions">
    <button type="button" data-run="short">Short toy fold</button>
    <button type="button" data-run="long">Long toy fold</button>
    <button type="button" data-run="lysozyme">Lysozyme pictures</button>
    <button type="button" data-run="compare">Compare maps</button>
    <button type="button" data-run="sickle">Sickle-cell picture</button>
    <button type="button" data-run="score">Transformer score</button>
    <button type="button" class="quiet" data-run="train">Train again</button>
  </div>
  <p class="note">Transformer score shows the saved test. Train again is the only button that trains.</p>
  <p id="status"></p>
  <div id="out"></div>
</main>
<script>
const buttons = document.querySelectorAll("button[data-run]");
const loadForm = document.querySelector("#load");
const loadButton = loadForm.querySelector("button");
const status = document.querySelector("#status");
const out = document.querySelector("#out");

function show(data) {
  out.replaceChildren();
  const p = document.createElement("p");
  p.textContent = data.text || "";
  out.appendChild(p);
  (data.images || []).forEach((image) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = image.src;
    img.alt = image.alt || "";
    fig.appendChild(img);
    if (image.caption) {
      const cap = document.createElement("figcaption");
      cap.textContent = image.caption;
      fig.appendChild(cap);
    }
    out.appendChild(fig);
  });
  if (data.frame) {
    const frame = document.createElement("iframe");
    frame.className = "viewer";
    frame.src = data.frame.href;
    frame.title = data.frame.label || "3D view";
    out.appendChild(frame);
  }
  if (data.link) {
    const a = document.createElement("a");
    a.href = data.link.href;
    a.textContent = data.link.label;
    a.target = "_blank";
    out.appendChild(a);
  }
}

async function run(name, body) {
  buttons.forEach((button) => { button.disabled = true; });
  loadButton.disabled = true;
  status.textContent = name === "train"
    ? "Training again. This takes a little while."
    : name === "protein"
      ? "Fetching the structure…"
      : "Working…";
  out.replaceChildren();
  try {
    const res = await fetch("/run/" + name, {
      method: "POST",
      headers: body ? {"Content-Type": "application/json"} : undefined,
      body: body ? JSON.stringify(body) : "",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "That did not work.");
    show(data);
    status.textContent = "Done.";
  } catch (err) {
    status.textContent = err.message || "That did not work.";
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
    loadButton.disabled = false;
  }
}

buttons.forEach((button) => {
  button.addEventListener("click", () => run(button.dataset.run));
});
loadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  run("protein", {code: new FormData(loadForm).get("code") || ""});
});
</script>
</body>
</html>
"""


def saved_score() -> dict:
    """The published test numbers. This does not load the model or train."""
    payload = dict(SAVED_SCORE)
    payload["trained"] = False
    payload["text"] = _score_text(SAVED_SCORE, saved=True)
    return payload


def _score_text(metrics: dict, saved: bool) -> str:
    lead = (
        "Saved test. These numbers are from the writeup, not a new training run."
        if saved
        else "New training run, just finished."
    )
    return (
        f"{lead} "
        f"Average miss on the energy: {metrics['mae']:.4f}. "
        f"Guessing the training average ({_minus(f"{metrics['mean_energy']:.4f}")}) misses by "
        f"{metrics['baseline_mae']:.4f}. "
        f"Low-energy means a score at or below {_minus(f"{metrics['threshold']:.0f}")}. "
        f"The model got that call right {metrics['accuracy']:.4f} of the time. "
        f"Always saying low-energy got {metrics['baseline_accuracy']:.4f}. "
        f"{int(metrics['n_train'])} chains to learn from, {int(metrics['n_test'])} held out."
    )


def _minus(value: int | float | str) -> str:
    return str(value).replace("-", "−")


def _file_url(name: str) -> str:
    path = ARTIFACTS / name
    stamp = int(path.stat().st_mtime) if path.is_file() else 0
    return f"/files/{name}?v={stamp}"


def _image(name: str, alt: str, caption: str) -> dict:
    return {"src": _file_url(name), "alt": alt, "caption": caption}


def run_short_fold() -> dict:
    from foldlab.cli import _exhaustive
    from foldlab.model import contacts
    from foldlab.sequences import DESIGNING_14

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = _exhaustive(DESIGNING_14, ARTIFACTS / "demo-short.gif", title="14.1")
    n_contacts = len(contacts(result.sequence, result.coords))
    return {
        "text": (
            f"Short chain. Every shape was checked. "
            f"Score {_minus(result.energy)}. "
            f"{n_contacts} sticky contacts. Lower is stickier."
        ),
        "images": [_image("demo-short.gif", "Short toy fold", "The short fold, drawn one bead at a time.")],
    }


def run_long_fold() -> dict:
    from foldlab.cli import DEMO_LONG_FRAME_MS, DEMO_LONG_HOLD_MS, _anneal
    from foldlab.model import contacts
    from foldlab.search import DEFAULT_ANNEAL_STEPS
    from foldlab.sequences import HP20

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = _anneal(
        HP20,
        seed=0,
        steps=DEFAULT_ANNEAL_STEPS,
        gif=ARTIFACTS / "demo-long.gif",
        title="HP-20",
        frame_ms=DEMO_LONG_FRAME_MS,
        hold_ms=DEMO_LONG_HOLD_MS,
    )
    n_contacts = len(contacts(result.sequence, result.coords))
    return {
        "text": (
            f"Longer chain. Cooled search, seed {result.seed}, {result.steps} steps. "
            f"Score {_minus(result.energy)}. {n_contacts} sticky contacts. "
            f"This is the best score that search saw."
        ),
        "images": [_image("demo-long.gif", "Long toy fold", "The longer fold, then a hold on the best shape.")],
    }


def _pictures() -> dict:
    from foldlab.phase2 import run_phase2

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return run_phase2(ARTIFACTS)


def run_lysozyme() -> dict:
    lysozyme = _pictures()["lysozyme"]
    return {
        "text": (
            f"Hen egg lysozyme. {lysozyme['contacts']} pairs sit close. "
            f"{lysozyme['n_h']} oily residues and {lysozyme['n_p']} watery ones. "
            f"Oily residues average {lysozyme['h_centroid']:.2f} Å from the center; "
            f"watery ones average {lysozyme['p_centroid']:.2f} Å."
        ),
        "images": [
            _image(
                "lysozyme-contact-map.png",
                "Contact map",
                "Contact map. Dark marks mean those parts sit close together.",
            ),
            _image(
                "lysozyme-hydrophobicity.png",
                "Oiliness chart",
                "Oiliness chart. Higher means oilier.",
            ),
        ],
        "link": {"href": _file_url("lysozyme-hp-core.html"), "label": "Open the 3D view"},
    }


def run_compare() -> dict:
    pictures = _pictures()
    lysozyme = pictures["lysozyme"]
    myoglobin = pictures["myoglobin"]
    return {
        "text": (
            "Same rule for both: backbone atoms within 8 Å, at least 4 residues apart. "
            f"Lysozyme, code 1LYZ: {lysozyme['residues']} residues, {lysozyme['contacts']} close pairs, "
            f"{lysozyme['near']} of them only 4 to 6 residues apart. "
            f"Myoglobin, code 1MBN: {myoglobin['residues']} residues, {myoglobin['contacts']} close pairs, "
            f"{myoglobin['near']} of them only 4 to 6 residues apart. "
            "Myoglobin is almost all helix. Lysozyme mixes helix and sheet."
        ),
        "images": [
            _image(
                "contact-compare.png",
                "Lysozyme and myoglobin contact maps",
                "Lysozyme on the left, myoglobin on the right.",
            ),
            _image("lysozyme-contact-map.png", "Lysozyme contact map", "Lysozyme, 1LYZ."),
            _image("myoglobin-contact-map.png", "Myoglobin contact map", "Myoglobin, 1MBN."),
        ],
    }


def run_sickle() -> dict:
    sickle = _pictures()["sickle"]
    site = (
        "glutamate"
        if sickle["beta_b"] == sickle["beta_d"] == "GLU"
        else f"{sickle['beta_b']} and {sickle['beta_d']}"
    )
    return {
        "text": (
            f"Hemoglobin. Both beta chains still have {site} at position 6. "
            f"Glutamate scores {_minus(sickle['glu'])} (watery). "
            f"Valine would score {sickle['val']} (oily). "
            f"The picture only changes the label."
        ),
        "images": [
            _image(
                "sickle-cell-glu6val.png",
                "Sickle-cell picture",
                "Sickle-cell picture. The marked spot is beta position 6.",
            )
        ],
    }


def _protein_url(folder: str, name: str) -> str:
    path = ARTIFACTS / "proteins" / folder / name
    stamp = int(path.stat().st_mtime) if path.is_file() else 0
    return f"/proteins/{folder}/{name}?v={stamp}"


def _distance(value: float) -> str:
    if value != value:
        return "no residues of that kind"
    return f"{value:.2f} Å from the center"


def run_protein(raw: bytes) -> dict:
    from foldlab.intake import build_protein

    try:
        payload = json.loads(raw.decode() or "{}")
    except json.JSONDecodeError as exc:
        from foldlab.intake import ProteinError

        raise ProteinError("Type a protein code, such as 1LYZ.") from exc
    info = build_protein(str(payload.get("code", "")), ARTIFACTS / "proteins")
    name = info["name"] or "This chain"
    others = ""
    if info["other_chains"]:
        listed = ", ".join(info["other_chains"])
        example = info["other_chains"][0]
        others = (
            f" Other chains in the file: {listed}. "
            f"Load one with {info['code']} {example}."
        )
    view = _protein_url(info["folder"], "view.html")
    return {
        "text": (
            f"{name}. Code {info['code']}, chain {info['chain']}. "
            f"{info['residues']} residues, {info['contacts']} close pairs. "
            f"{info['n_h']} oily and {info['n_p']} watery. "
            f"Oily residues average {_distance(info['h_centroid'])}; "
            f"watery ones average {_distance(info['p_centroid'])}. "
            f"Helix {info['n_helix']}, sheet {info['n_sheet']}, coil {info['n_coil']}. "
            "In the 3D view, rust is oily and blue is watery."
            + others
        ),
        "images": [
            _image_url(
                _protein_url(info["folder"], "contact.png"),
                "Contact map",
                "Contact map. Dark marks mean those parts sit close together.",
            ),
            _image_url(
                _protein_url(info["folder"], "oiliness.png"),
                "Oiliness chart",
                "Oiliness chart. Higher means oilier.",
            ),
        ],
        "frame": {"href": view, "label": "3D view"},
        "link": {"href": view, "label": "Open the 3D view"},
    }


def _image_url(src: str, alt: str, caption: str) -> dict:
    return {"src": src, "alt": alt, "caption": caption}


def run_train() -> dict:
    from foldlab.learn import generate_dataset, load_dataset, train_and_evaluate

    path = generate_dataset()
    metrics = train_and_evaluate(load_dataset(path))
    return {"trained": True, "text": _score_text(metrics, saved=False), **metrics}


ACTIONS = {
    "short": run_short_fold,
    "long": run_long_fold,
    "lysozyme": run_lysozyme,
    "compare": run_compare,
    "sickle": run_sickle,
    "score": saved_score,
    "train": run_train,
    "protein": run_protein,
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode()
            self._send(200, "text/html; charset=utf-8", body)
            return
        if path == "/favicon.ico":
            self._send(204, "text/plain", b"")
            return
        if path.startswith("/files/"):
            name = path.removeprefix("/files/")
            if name not in FILES:
                self._send(404, "text/plain; charset=utf-8", b"No such file.\n")
                return
            file_path = ARTIFACTS / name
            if not file_path.is_file():
                self._send(404, "text/plain; charset=utf-8", b"That picture has not been written yet.\n")
                return
            self._send(200, FILES[name], file_path.read_bytes())
            return
        match = _PROTEIN_PATH.fullmatch(path)
        if match:
            folder, name = match.groups()
            file_path = ARTIFACTS / "proteins" / folder / name
            if not file_path.is_file():
                self._send(404, "text/plain; charset=utf-8", b"That picture has not been written yet.\n")
                return
            self._send(200, _PROTEIN_TYPES[name], file_path.read_bytes())
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found.\n")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        path = urlparse(self.path).path
        action = path.removeprefix("/run/") if path.startswith("/run/") else ""
        if action not in ACTIONS:
            self._json(404, {"error": "No such button."})
            return
        from foldlab.intake import ProteinError

        try:
            with _JOB:
                payload = ACTIONS[action](raw) if action == "protein" else ACTIONS[action]()
        except ProteinError as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "That did not work."})
            return
        self._json(200, payload)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, "application/json", json.dumps(payload).encode())

    def log_message(self, fmt: str, *args) -> None:
        print("[screen]", fmt % args)


def make_httpd(port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, port), Handler)


def main() -> None:
    httpd = make_httpd(PORT)
    print(f"FoldLab screen at http://{HOST}:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
