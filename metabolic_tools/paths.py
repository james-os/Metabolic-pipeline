"""
A registry of where the datasets actually live, so nothing has to carry an absolute path.

The same dataset sits in different places on the laptop and on CREATE, and CREATE keeps
the real data in group scratch rather than home. Rather than remember either, register a
dataset once per machine and refer to it by short name from then on:

    python -m metabolic_tools.paths                          # what is registered, and what exists here
    python -m metabolic_tools.paths --add kolla_e16_cells /scratch/.../kolla_e16.h5ad --host create
    python -m metabolic_tools.paths --resolve kolla_e16_cells

    from metabolic_tools.paths import resolve
    adata_path = resolve("kolla_e16_cells")

Resolution tries every registered path for a name and returns the first that exists, so
the 'host' column is a label for humans rather than something the code matches on. Paths
may contain environment variables and ~, which is the intended way to write CREATE paths:
$MET/data/kolla_e16.h5ad stays correct if the scratch project moves.

The registry is a TSV in the repo, so it is version controlled and doubles as the record
of what each file is -- shape, cell-type column, and where it came from.
"""

import argparse
import csv
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REGISTRY = os.path.join(DATA_DIR, "dataset_registry.tsv")
FIELDS = ["name", "host", "path", "celltype_col", "gene_column", "n_obs", "n_vars", "notes"]


def expand(path):
    """Expand ~ and $VARS, so a registered path can be written relative to $MET."""
    return os.path.expanduser(os.path.expandvars(path))


def load(registry=REGISTRY):
    if not os.path.exists(registry):
        return []
    with open(registry, encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh, delimiter="\t") if row.get("name")]


def entries(name, registry=REGISTRY):
    """Every registered row for a dataset, in file order."""
    rows = [r for r in load(registry) if r["name"] == name]
    if not rows:
        known = sorted({r["name"] for r in load(registry)})
        raise KeyError(f"{name!r} is not registered. Known datasets: {', '.join(known) or 'none'}")
    return rows


def resolve(name, registry=REGISTRY):
    """The first registered path for this dataset that exists on this machine."""
    tried = []
    for row in entries(name, registry):
        path = expand(row["path"])
        if os.path.exists(path):
            return path
        tried.append(f"  [{row['host']}] {row['path']}")
    raise FileNotFoundError(
        f"{name!r} is registered but none of its paths exist here:\n" + "\n".join(tried)
        + f"\nRegister this machine's copy with:\n"
          f"  python -m metabolic_tools.paths --add {name} <path> --host <label>")


def info(name, registry=REGISTRY):
    """The registry row backing resolve(), for the cell-type and gene columns."""
    for row in entries(name, registry):
        if os.path.exists(expand(row["path"])):
            return row
    return entries(name, registry)[0]


def add(name, path, host="", celltype_col="", gene_column="gene_symbol",
        notes="", registry=REGISTRY, describe=True):
    """
    Register a dataset. Shape is read off the file when it is readable, which is what
    makes the registry useful later -- the row says what the file is, not just where.
    """
    n_obs = n_vars = ""
    real = expand(path)
    if describe and os.path.exists(real):
        try:
            import anndata
            adata = anndata.read_h5ad(real, backed="r")
            n_obs, n_vars = str(adata.n_obs), str(adata.n_vars)
            if not celltype_col:
                for candidate in ("cell_type", "celltype", "majority_celltype", "louvain"):
                    if candidate in adata.obs.columns:
                        celltype_col = candidate
                        break
        except Exception as exc:                       # not an h5ad, or unreadable
            notes = (notes + f" (not described: {type(exc).__name__})").strip()

    rows = load(registry)
    rows = [r for r in rows if not (r["name"] == name and r["host"] == host)]
    rows.append({"name": name, "host": host, "path": path, "celltype_col": celltype_col,
                 "gene_column": gene_column, "n_obs": n_obs, "n_vars": n_vars, "notes": notes})
    rows.sort(key=lambda r: (r["name"], r["host"]))
    with open(registry, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"registered {name} [{host or 'no host label'}] -> {path}"
          + (f"  ({n_obs} x {n_vars})" if n_obs else "  (file not present here)"))
    return rows


def show(registry=REGISTRY):
    rows = load(registry)
    if not rows:
        print("nothing registered yet")
        return rows
    width = max(len(r["name"]) for r in rows)
    print(f"{'':2s} {'name':{width}s}  {'host':8s}  {'shape':>16s}  path")
    for row in rows:
        here = os.path.exists(expand(row["path"]))
        shape = f"{row['n_obs']} x {row['n_vars']}" if row["n_obs"] else ""
        print(f"{'ok' if here else '--':2s} {row['name']:{width}s}  {row['host']:8s}  "
              f"{shape:>16s}  {row['path']}")
    print("\nok = present on this machine, -- = registered elsewhere")
    return rows


#: public name for the registration entry point
register_dataset = add


def main():
    parser = argparse.ArgumentParser(description="Where the datasets live")
    parser.add_argument("--add", nargs=2, metavar=("NAME", "PATH"))
    parser.add_argument("--host", default="", help="label for this machine, e.g. create or laptop")
    parser.add_argument("--celltype-col", default="")
    parser.add_argument("--gene-column", default="gene_symbol")
    parser.add_argument("--notes", default="")
    parser.add_argument("--resolve", metavar="NAME")
    args = parser.parse_args()
    if args.add:
        add(args.add[0], args.add[1], args.host, args.celltype_col, args.gene_column, args.notes)
    elif args.resolve:
        print(resolve(args.resolve))
    else:
        show()


if __name__ == "__main__":
    main()
