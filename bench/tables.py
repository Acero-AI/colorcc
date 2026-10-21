"""Print the Markdown tables of results/README.md from the result files, so the tables are the data.

    python bench/tables.py > /tmp/tables.md

Every number in the blog post (https://blogs.fablnx.com/posts/every-colour/) comes from these files too.
"""
import json
import pathlib

R = pathlib.Path(__file__).resolve().parent.parent / "results"
SWEEPS = {"Apple M5 Pro": "m5/m5pro_sheet.json", "RTX 3070": "rtx3070/sheet.json", "A100 SXM4": "a100/sheet.json"}
COPIES = {"Apple M5 Pro": "m5/e2e.json", "RTX 3070": "rtx3070/e2e.json", "A100 SXM4": "a100/e2e.json"}
LIBS = {"A100 SXM4": ["a100/nvidia.json"], "RTX 3070": ["rtx3070/nvidia.json"]}
IDEAS = [("plain, first three stages", "gpu:base", "joins:base"), ("+ warp vote (runs)", "gpu:+runs", "joins:+runs"),
         ("+ skip joins", "gpu:+skip", "joins:+skip"), ("+ path halving", "gpu:+halve", "joins:+halve"),
         ("all three", "gpu:all-packed", "joins:all-packed"), ("+ numbering (the full call)", "gpu:all+n", None)]


def rows(path):
    d = json.loads((R / path).read_text())
    return d["rows"] if isinstance(d, dict) else d


def ms(t):
    return "–" if t is None else f"{t * 1e3:.2f} ms" if t < 0.01 else f"{t * 1e3:.1f} ms" if t < 1 else f"{t:.2f} s"


def mb(b):
    return "–" if not b else f"{b / 1e6:.1f} MB" if b < 1e9 else f"{b / 1e9:.2f} GB"


def table(head, body):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body])


def at800(path, paper=True):
    return next(r for r in rows(path) if r["dpi"] == 800 and r["ignore_white"] != paper)


def libs(files):
    """nvidia.py rows by file name; a later file's measurement replaces an earlier one, an error never replaces one."""
    got = {}
    for f in files:
        for r in rows(f):
            g = got.setdefault(r["file"], {"dpi": int(r["file"].split(".")[-2]), "mpx": r["H"] * r["W"] / 1e6})
            for k, v in r.items():
                if isinstance(v, dict) and ("time" in v or k not in g):
                    g[k] = v
    return sorted(got.values(), key=lambda g: g["dpi"])


def main():
    out = []
    head = ["GPU", "time", "with copies in and out"]
    body = []
    for gpu, f in SWEEPS.items():
        r, e = at800(f), next(x for x in rows(COPIES[gpu]) if x["file"].endswith(".800.npy"))
        body.append([gpu, ms(r["gpu:all+n"]), ms(e["reuse"])])
    out += ["### The full sheet: 800 dpi, 806 million pixels, paper labelled", "", table(head, body), ""]
    for gpu, f in SWEEPS.items():
        r = at800(f)
        out += [f"### Each idea, 800 dpi, {gpu}", "",
                table(["", "time", "joins"], [[n, ms(r.get(t)), f"{r[j] / 1e9:.2f} billion" if j and r.get(j) else ""] for n, t, j in IDEAS]), ""]
    for gpu, files in LIBS.items():
        body = []
        for g in libs(files):
            o, c, n = (g.get(k, {}) for k in ("ours", "cucim", "npp_raw"))
            ratio = lambda a, b: f"{a['time'] / b['time']:.1f}x" if "time" in a and "time" in b else "–"
            body.append([str(g["dpi"]), f"{g['mpx']:.0f}", ms(o.get("time")), ms(c.get("time")), ratio(c, o),
                         ms(n.get("time")), ratio(n, o), mb(o.get("peak")), mb(c.get("peak")), mb(n.get("peak"))])
        out += [f"### Against NVIDIA's libraries, {gpu}", "",
                table(["dpi", "MP", "ours", "cuCIM", "cuCIM / ours", "NPP (wrong labels)", "NPP / ours", "memory: ours", "cuCIM", "NPP"], body), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
