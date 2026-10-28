"""Phase 2 figures: lysozyme contacts, hydropathy, H/P view, and the β6 comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from foldlab.kd import PROFILE_WINDOW, hp_class, kd_score, window_profile
from foldlab.structure import (
    CONTACT_CUTOFF,
    MIN_SEPARATION,
    cached_pdb,
    ca_contacts,
    contact_matrix,
    load_chains,
    mean_centroid_distance,
)
from foldlab.viewer import H_COLOR, P_COLOR, write_hp_view

PAPER = "#fffbf5"
INK = "#1c1917"
MUTED = "#57534e"
LYSOZYME_ID = "1LYZ"
MYOGLOBIN_ID = "1MBN"
HEMOGLOBIN_ID = "2HHB"


def _chain_pdb(path: Path, chain_id: str) -> str:
    kept = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM") and len(line) > 21 and line[21] == chain_id:
            kept.append(line)
    kept.append("END")
    return "\n".join(kept) + "\n"


def _style(ax) -> None:
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#e7e5e4")


def write_contact_map(residues, pairs, path: Path, title: str | None = None) -> int:
    matrix = contact_matrix(len(residues), pairs)
    n = len(residues)
    fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=140)
    fig.patch.set_facecolor(PAPER)
    ax.imshow(
        matrix,
        cmap=ListedColormap([PAPER, H_COLOR]),
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
        extent=(0.5, n + 0.5, n + 0.5, 0.5),
    )
    ax.set_title(
        title
        or f"1LYZ Cα contacts, ≤ {CONTACT_CUTOFF:.0f} Å, |i−j| ≥ {MIN_SEPARATION}",
        color=INK,
        pad=10,
    )
    ax.set_xlabel("residue", color=MUTED)
    ax.set_ylabel("residue", color=MUTED)
    _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return len(pairs)


def write_hydrophobicity(residues, path: Path) -> dict:
    scores = np.array([kd_score(res.name1) for res in residues])
    classes = [hp_class(res.name1) for res in residues]
    smooth = window_profile(scores, PROFILE_WINDOW)
    h_mean, p_mean = mean_centroid_distance(residues, classes)
    colors = [H_COLOR if klass == "H" else P_COLOR for klass in classes]
    ss_color = {"H": H_COLOR, "E": P_COLOR, "C": "#e7e5e4"}
    strip = np.array([[{"H": 2, "E": 1, "C": 0}[res.ss] for res in residues]])

    fig, (ax, axss) = plt.subplots(
        2,
        1,
        figsize=(11.2, 4.8),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [5, 0.55], "hspace": 0.08},
    )
    fig.patch.set_facecolor(PAPER)
    x = np.arange(1, len(residues) + 1)
    ax.bar(x, scores, color=colors, width=1.0, linewidth=0)
    ax.plot(x, smooth, color=INK, linewidth=1.4, label=f"{PROFILE_WINDOW}-residue window")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Kyte–Doolittle", color=MUTED)
    ax.set_title(
        f"1LYZ hydropathy    H {h_mean:.2f} Å from centroid    P {p_mean:.2f} Å",
        color=INK,
        pad=10,
    )
    ax.legend(frameon=False, loc="upper right")
    _style(ax)
    axss.imshow(
        strip,
        cmap=ListedColormap([ss_color["C"], ss_color["E"], ss_color["H"]]),
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=2,
        extent=(0.5, len(residues) + 0.5, -0.5, 0.5),
    )
    axss.set_yticks([0])
    axss.set_yticklabels(["SS"])
    axss.set_xlabel("residue    rust helix    blue sheet    gray coil    from HELIX/SHEET", color=MUTED)
    axss.tick_params(axis="x", colors=MUTED)
    for spine in axss.spines.values():
        spine.set_color("#e7e5e4")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_h": classes.count("H"),
        "n_p": classes.count("P"),
        "h_centroid": h_mean,
        "p_centroid": p_mean,
        "n_helix": sum(res.ss == "H" for res in residues),
        "n_sheet": sum(res.ss == "E" for res in residues),
        "n_coil": sum(res.ss == "C" for res in residues),
    }


def write_lysozyme(out_dir: Path) -> dict:
    path = cached_pdb(LYSOZYME_ID)
    chains = load_chains(path)
    residues = chains["A"]
    pairs = ca_contacts(residues)
    contacts = write_contact_map(residues, pairs, out_dir / "lysozyme-contact-map.png")
    hydro = write_hydrophobicity(residues, out_dir / "lysozyme-hydrophobicity.png")
    classes = [(res.resseq, hp_class(res.name1)) for res in residues]
    write_hp_view(
        out_dir / "lysozyme-hp-core.html",
        _chain_pdb(path, "A"),
        "A",
        classes,
        "1LYZ hen egg-white lysozyme",
    )
    hydro["contacts"] = contacts
    hydro["residues"] = len(residues)
    hydro.update(_span_counts(residues, pairs))
    return hydro


def _span_counts(residues, pairs) -> dict[str, int]:
    """How far apart the two residues are along the chain."""
    near = mid = far = 0
    for i, j in pairs:
        sep = abs(residues[i].resseq - residues[j].resseq)
        if sep <= 6:
            near += 1
        elif sep <= 20:
            mid += 1
        else:
            far += 1
    return {"near": near, "mid": mid, "far": far}


def write_myoglobin(out_dir: Path) -> dict:
    """Sperm whale myoglobin, same Cα rule as lysozyme."""
    path = cached_pdb(MYOGLOBIN_ID)
    residues = load_chains(path)["A"]
    pairs = ca_contacts(residues)
    contacts = write_contact_map(
        residues,
        pairs,
        out_dir / "myoglobin-contact-map.png",
        title=f"1MBN Cα contacts, ≤ {CONTACT_CUTOFF:.0f} Å, |i−j| ≥ {MIN_SEPARATION}",
    )
    return {
        "residues": len(residues),
        "contacts": contacts,
        "n_helix": sum(res.ss == "H" for res in residues),
        "n_sheet": sum(res.ss == "E" for res in residues),
        "n_coil": sum(res.ss == "C" for res in residues),
        **_span_counts(residues, pairs),
    }


def write_contact_comparison(out_dir: Path) -> None:
    """Lysozyme and myoglobin side by side, one contact rule."""
    panels = (
        (LYSOZYME_ID, "lysozyme"),
        (MYOGLOBIN_ID, "myoglobin"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), dpi=140)
    fig.patch.set_facecolor(PAPER)
    for ax, (pdb_id, name) in zip(axes, panels):
        residues = load_chains(cached_pdb(pdb_id))["A"]
        pairs = ca_contacts(residues)
        matrix = contact_matrix(len(residues), pairs)
        n = len(residues)
        ax.imshow(
            matrix,
            cmap=ListedColormap([PAPER, H_COLOR]),
            interpolation="nearest",
            origin="upper",
            vmin=0,
            vmax=1,
            extent=(0.5, n + 0.5, n + 0.5, 0.5),
        )
        ax.set_title(f"{pdb_id} {name}\n{n} residues, {len(pairs)} contacts", color=INK, pad=8)
        ax.set_xlabel("residue", color=MUTED)
        ax.set_ylabel("residue", color=MUTED)
        _style(ax)
    fig.suptitle(
        f"Cα ≤ {CONTACT_CUTOFF:.0f} Å, |i−j| ≥ {MIN_SEPARATION}",
        color=INK,
    )
    fig.tight_layout()
    dest = out_dir / "contact-compare.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def _patch(residues_by_chain, chain_id: str, resseq: int, cutoff: float = CONTACT_CUTOFF):
    target = next(res for res in residues_by_chain[chain_id] if res.resseq == resseq)
    if target.name3 != "GLU":
        raise ValueError(f"{chain_id} {resseq} is {target.name3}, expected GLU")
    near = []
    for chain in residues_by_chain.values():
        for res in chain:
            if res.chain == target.chain and res.resseq == target.resseq:
                continue
            dist = float(np.linalg.norm(res.ca - target.ca))
            if dist <= cutoff:
                near.append(res)
    return target, near


def write_sickle(out_dir: Path) -> dict:
    path = cached_pdb(HEMOGLOBIN_ID)
    chains = load_chains(path)
    site_b, patch_b = _patch(chains, "B", 6)
    site_d, patch_d = _patch(chains, "D", 6)
    all_res = [res for chain in chains.values() for res in chain]
    coords = np.vstack([res.ca for res in all_res])

    fig = plt.figure(figsize=(11.4, 5.8), dpi=140)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=6, c="#d6d3d1", depthshade=False)
    for chain_id, color in (("B", "#9a3412"), ("D", "#075985")):
        chain_xyz = np.vstack([res.ca for res in chains[chain_id]])
        ax.plot(chain_xyz[:, 0], chain_xyz[:, 1], chain_xyz[:, 2], color=color, linewidth=1.2)
    patch_xyz = np.vstack([res.ca for res in patch_b]) if patch_b else np.zeros((0, 3))
    if len(patch_xyz):
        patch_colors = [H_COLOR if hp_class(res.name1) == "H" else P_COLOR for res in patch_b]
        ax.scatter(
            patch_xyz[:, 0],
            patch_xyz[:, 1],
            patch_xyz[:, 2],
            s=46,
            c=patch_colors,
            depthshade=False,
            edgecolors="white",
            linewidths=0.4,
        )
    for site in (site_b, site_d):
        ax.scatter(
            [site.ca[0]],
            [site.ca[1]],
            [site.ca[2]],
            s=140,
            c="#7c2d12",
            depthshade=False,
            marker="o",
            edgecolors="white",
            linewidths=0.6,
        )
    center = coords.mean(axis=0)
    radius = np.abs(coords - center).max()
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("2HHB    β6 Glu marked", color=INK, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-70)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PAPER)
        axis.pane.set_edgecolor("#e7e5e4")

    ax2 = fig.add_subplot(1, 2, 2)
    labels = ["β6 Glu\n(in 2HHB)", "Val\n(relabel)"]
    values = [kd_score("E"), kd_score("V")]
    ax2.bar(labels, values, color=[P_COLOR, H_COLOR], width=0.62)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_ylabel("Kyte–Doolittle", color=MUTED)
    n_h = sum(hp_class(res.name1) == "H" for res in patch_b)
    n_p = len(patch_b) - n_h
    # Glu is P, so relabeling this site as Val adds one H and does not change the neighbors.
    ax2.set_title(
        f"βB 6 patch, Cα ≤ {CONTACT_CUTOFF:.0f} Å: {n_h} H and {n_p} P neighbors\n"
        f"relabeling β6 as Val makes the site itself H",
        color=INK,
        fontsize=11,
        pad=10,
    )
    _style(ax2)
    fig.text(
        0.5,
        0.02,
        "Published 2HHB coordinates. The Glu→Val change is a class label on β6 (chains B and D), not a rebuilt side chain.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    dest = out_dir / "sickle-cell-glu6val.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return {
        "beta_b": site_b.name3,
        "beta_d": site_d.name3,
        "patch_b": len(patch_b),
        "patch_d": len(patch_d),
        "patch_b_h": n_h,
        "patch_b_p": n_p,
        "glu": kd_score("E"),
        "val": kd_score("V"),
    }


def run_phase2(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    lysozyme = write_lysozyme(out_dir)
    myoglobin = write_myoglobin(out_dir)
    write_contact_comparison(out_dir)
    sickle = write_sickle(out_dir)
    print(f"1LYZ residues:  {lysozyme['residues']}")
    print(f"contacts:       {lysozyme['contacts']}  (Cα ≤ {CONTACT_CUTOFF:.0f} Å, |i−j| ≥ {MIN_SEPARATION})")
    print(f"H / P:          {lysozyme['n_h']} / {lysozyme['n_p']}")
    print(
        f"centroid Å:     H {lysozyme['h_centroid']:.2f}    P {lysozyme['p_centroid']:.2f}"
    )
    print(
        f"secondary:      helix {lysozyme['n_helix']}    sheet {lysozyme['n_sheet']}    coil {lysozyme['n_coil']}"
    )
    print(
        f"1MBN residues:  {myoglobin['residues']}    contacts {myoglobin['contacts']}"
    )
    print(
        f"1MBN spans:     4–6 {myoglobin['near']}    7–20 {myoglobin['mid']}    >20 {myoglobin['far']}"
    )
    print(
        f"1MBN secondary: helix {myoglobin['n_helix']}    sheet {myoglobin['n_sheet']}    coil {myoglobin['n_coil']}"
    )
    print(
        f"2HHB β6:        chain B {sickle['beta_b']}    chain D {sickle['beta_d']}"
    )
    print(
        f"βB6 patch:      {sickle['patch_b']} neighbors    {sickle['patch_b_h']} H    {sickle['patch_b_p']} P"
    )
    print(f"KD Glu / Val:   {sickle['glu']} / {sickle['val']}")
    for name in (
        "lysozyme-contact-map.png",
        "lysozyme-hydrophobicity.png",
        "lysozyme-hp-core.html",
        "myoglobin-contact-map.png",
        "contact-compare.png",
        "sickle-cell-glu6val.png",
    ):
        print(f"wrote:          {out_dir / name}")
    return {"lysozyme": lysozyme, "myoglobin": myoglobin, "sickle": sickle}
