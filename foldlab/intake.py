"""Load any published structure by its 4-character code.

The contact rule, the oiliness scale, and the 3D coloring are the same ones
used for lysozyme. The file is cached under data/ after the first download.
"""

from __future__ import annotations

from pathlib import Path
import re
import socket

from foldlab.kd import hp_class
from foldlab.phase2 import write_contact_map, write_hydrophobicity
from foldlab.structure import (
    CONTACT_CUTOFF,
    MIN_SEPARATION,
    cached_pdb,
    ca_contacts,
    load_chains,
    mean_centroid_distance,
)
from foldlab.viewer import write_hp_view

_CODE = re.compile(r"^([0-9][A-Za-z0-9]{3})(?:[\s:,]+([A-Za-z0-9]))?$")
MAX_BYTES = 25_000_000
MAX_RESIDUES = 2_000


class ProteinError(ValueError):
    """A code the screen can explain in one sentence."""


def parse_code(text: str) -> tuple[str, str | None]:
    """Return (PDB code, chain or None). Chain may follow a space or colon."""
    match = _CODE.fullmatch(" ".join(text.strip().split()))
    if not match:
        raise ProteinError(
            "Use a 4-character protein code, such as 1LYZ. "
            "Add a chain like 2HHB B if you need one."
        )
    code = match.group(1).upper()
    chain = match.group(2).upper() if match.group(2) else None
    return code, chain


def _molecule_name(text: str, chain: str) -> str:
    """The COMPND name for this chain, when the file records one."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("COMPND"):
            continue
        body = line[10:].strip()
        if body.startswith("MOL_ID:"):
            if current:
                blocks.append(current)
            current = {}
            continue
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        current[key.strip()] = value.strip().rstrip(";").strip()
    if current:
        blocks.append(current)
    for block in blocks:
        chains = [item.strip() for item in block.get("CHAIN", "").split(",") if item.strip()]
        if chain in chains and block.get("MOLECULE"):
            return block["MOLECULE"].title()
    for block in blocks:
        if block.get("MOLECULE"):
            return block["MOLECULE"].title()
    return ""


def _chain_pdb(path: Path, chain_id: str) -> str:
    kept = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and len(line) > 21 and line[21] == chain_id:
            kept.append(line)
    kept.append("END")
    return "\n".join(kept) + "\n"


def _fetch(code: str) -> Path:
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(45)
    try:
        path = cached_pdb(code)
    except Exception as exc:
        raise ProteinError("No structure came back for that code.") from exc
    finally:
        socket.setdefaulttimeout(previous)
    if path.stat().st_size > MAX_BYTES:
        raise ProteinError("That structure is too big for this screen.")
    text = path.read_text(errors="replace")
    if "\nATOM" not in text and not text.startswith("ATOM"):
        path.unlink(missing_ok=True)
        raise ProteinError("No structure came back for that code.")
    return path


def _finite(value: float) -> str | None:
    if value != value:
        return None
    return f"{value:.2f}"


def build_protein(text: str, out_root: Path) -> dict:
    """Download the structure if needed and write the three views for one chain."""
    code, requested = parse_code(text)
    path = _fetch(code)
    chains = load_chains(path)
    if not chains:
        raise ProteinError("That file has no protein chain.")
    if requested is None:
        chain = max(chains, key=lambda chain_id: len(chains[chain_id]))
    elif requested not in chains:
        names = ", ".join(sorted(chains))
        raise ProteinError(f"No chain {requested}. Chains in this file: {names}.")
    else:
        chain = requested
    residues = chains[chain]
    if len(residues) > MAX_RESIDUES:
        raise ProteinError("That chain is too long for this screen.")

    pairs = ca_contacts(residues)
    classes = [hp_class(res.name1) for res in residues]
    h_mean, p_mean = mean_centroid_distance(residues, classes)
    folder = f"{code.lower()}-{chain.lower()}"
    dest = Path(out_root) / folder
    dest.mkdir(parents=True, exist_ok=True)
    rule = f"≤ {CONTACT_CUTOFF:.0f} Å, |i−j| ≥ {MIN_SEPARATION}"
    contacts = write_contact_map(
        residues,
        pairs,
        dest / "contact.png",
        title=f"{code} chain {chain} Cα contacts, {rule}",
    )
    h_bit = _finite(h_mean)
    p_bit = _finite(p_mean)
    oil_title = f"{code} chain {chain} oiliness"
    if h_bit and p_bit:
        oil_title += f"    oily {h_bit} Å from center    watery {p_bit} Å"
    hydro = write_hydrophobicity(residues, dest / "oiliness.png", title=oil_title)
    write_hp_view(
        dest / "view.html",
        _chain_pdb(path, chain),
        chain,
        [(res.resseq, hp_class(res.name1)) for res in residues],
        f"{code} chain {chain}",
    )
    others = sorted(chain_id for chain_id in chains if chain_id != chain)
    return {
        "code": code,
        "chain": chain,
        "name": _molecule_name(path.read_text(errors="replace"), chain),
        "residues": len(residues),
        "contacts": contacts,
        "n_h": hydro["n_h"],
        "n_p": hydro["n_p"],
        "h_centroid": h_mean,
        "p_centroid": p_mean,
        "n_helix": hydro["n_helix"],
        "n_sheet": hydro["n_sheet"],
        "n_coil": hydro["n_coil"],
        "other_chains": others,
        "folder": folder,
    }
