"""
Audit mitoMAMMALmod's gene-protein-reaction rules against a genome-scale model.

mitoMAMMAL inherits mitocore's deliberately reduced reaction set, so the question
this module answers is *not* "which reactions are missing" -- the model has to stay
compact. It is the narrower one: for the reactions mitoMAMMALmod already contains,
which genes does a genome-scale model assign that mitoMAMMALmod's own GPR leaves out?

That is the class of gap Ldha, Pfkl and Pfkp belonged to, and --validate re-finds
those three after deleting them, as a positive control.

Reference model: Mouse-GEM (Mouse1) from SysBioChalmers.
Gene namespaces are reconciled through MGI accessions, because mitoMAMMALmod names
genes with Ensembl IDs and Mouse-GEM with MGI symbols that have since been renamed
(Atp5a1 became Atp5f1a, and so on).

Usage
-----
    python -m metabolic_tools.model_gene_audit --out gene_audit.tsv
    python -m metabolic_tools.model_gene_audit --validate
"""

import argparse
import collections
import csv
import json
import os
import re
import urllib.request

from .gene_mapping import resolve_model_path

MOUSE_GEM_BASE = "https://raw.githubusercontent.com/SysBioChalmers/Mouse-GEM/main"
MGI_BASE = "https://www.informatics.jax.org/downloads/reports"
REFERENCE_FILES = {
    "Mouse-GEM.yml": MOUSE_GEM_BASE + "/model/Mouse-GEM.yml",
    "reactions.tsv": MOUSE_GEM_BASE + "/model/reactions.tsv",
    "metabolites.tsv": MOUSE_GEM_BASE + "/model/metabolites.tsv",
    "version.txt": MOUSE_GEM_BASE + "/version.txt",
    "MRK_List2.rpt": MGI_BASE + "/MRK_List2.rpt",
    "MGI_Gene_Model_Coord.rpt": MGI_BASE + "/MGI_Gene_Model_Coord.rpt",
}

# mitoMAMMAL has three compartments; Mouse-GEM has nine. 'i' (inner mitochondria)
# folds into 'm' because mitoMAMMAL keeps no separate intermembrane space.
COMPARTMENTS = {"Cytosol": "c", "Mitochondrion": "m", "external": "e"}
COMPARTMENT_COMPATIBLE = {"c": {"c"}, "m": {"m", "i"}, "e": {"e"}}

# Placeholders mitoMAMMAL writes instead of a gene, each meaning something
# different from an empty rule.
GPR_PLACEHOLDERS = {
    "unknown": "unknown",
    "non-enzymatic": "non_enzymatic",
    "n/a": "not_applicable",
}


# --------------------------------------------------------------------------- refs

def fetch_references(ref_dir, force=False):
    """Download Mouse-GEM and the MGI marker reports into ref_dir, skipping what is there."""
    os.makedirs(ref_dir, exist_ok=True)
    for name, url in REFERENCE_FILES.items():
        dest = os.path.join(ref_dir, name)
        if os.path.exists(dest) and not force:
            continue
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
    version = os.path.join(ref_dir, "version.txt")
    return open(version).read().strip() if os.path.exists(version) else "unknown"


# ---------------------------------------------------------------------- gene names

class GeneResolver:
    """Maps MGI symbols, symbol synonyms and Ensembl IDs onto a single MGI accession."""

    def __init__(self, ref_dir):
        self.sym_to_acc = {}
        self.syn_to_acc = collections.defaultdict(set)
        self.ens_to_acc = {}
        self.acc_to_sym = {}
        self.acc_to_ens = {}
        self.acc_to_name = {}

        with open(os.path.join(ref_dir, "MRK_List2.rpt"), encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 12 or parts[7] != "O":   # official markers only
                    continue
                acc, symbol, name, synonyms = parts[0], parts[6], parts[8], parts[11]
                self.sym_to_acc[symbol] = acc
                self.acc_to_sym[acc] = symbol
                self.acc_to_name[acc] = name
                for syn in synonyms.split("|"):
                    if syn:
                        self.syn_to_acc[syn].add(acc)

        with open(os.path.join(ref_dir, "MGI_Gene_Model_Coord.rpt"), encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 11:
                    continue
                acc, ensembl = parts[0], parts[10]
                if ensembl and ensembl != "null":
                    self.ens_to_acc[ensembl] = acc
                    self.acc_to_ens[acc] = ensembl

    def from_symbol(self, symbol):
        if symbol in self.sym_to_acc:
            return self.sym_to_acc[symbol]
        candidates = self.syn_to_acc.get(symbol)
        return next(iter(candidates)) if candidates and len(candidates) == 1 else None

    def from_ensembl(self, ensembl):
        return self.ens_to_acc.get(ensembl)

    def symbol(self, acc):
        return self.acc_to_sym.get(acc, acc)


def gpr_tokens(rule):
    return [t for t in re.findall(r"[A-Za-z0-9_\-\.]+", rule or "")
            if t.lower() not in ("and", "or")]


# ------------------------------------------------------------------- mouse-gem I/O

def load_mouse_gem(ref_dir):
    """Parse the RAVEN yaml plus the two annotation tables into reaction records."""
    reactions = _parse_yaml_reactions(os.path.join(ref_dir, "Mouse-GEM.yml"))
    rxn_table = {r["rxns"]: r for r in _read_tsv(os.path.join(ref_dir, "reactions.tsv"))}
    met_table = {r["mets"]: r for r in _read_tsv(os.path.join(ref_dir, "metabolites.tsv"))}
    return reactions, rxn_table, met_table


def _parse_yaml_reactions(path):
    """Hand-rolled reader for the reactions block. The layout is fully regular, and
    this keeps pyyaml out of the dependency list just to walk an 8 MB !!omap."""
    key_re = re.compile(r"^      - (\w+):\s*(.*)$")
    met_re = re.compile(r"^          - (\S+): (-?[\d.eE+]+)\s*$")
    sub_re = re.compile(r"^          - (.*)$")
    reactions, current, block = [], None, None
    with open(path, encoding="utf-8") as fh:
        in_reactions = False
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("- reactions:"):
                in_reactions = True
                continue
            if not in_reactions:
                continue
            if line.startswith("- "):            # next top-level block
                break
            if line.strip() == "- !!omap":
                current = {"metabolites": {}, "subsystem": []}
                reactions.append(current)
                block = None
                continue
            match = key_re.match(line)
            if match:
                key, value = match.group(1), match.group(2)
                if key in ("metabolites", "subsystem"):
                    block = key
                    if key == "subsystem" and value.strip():
                        current["subsystem"].append(_unquote(value))
                else:
                    block = None
                    current[key] = _unquote(value)
                continue
            if block == "metabolites":
                met = met_re.match(line)
                if met:
                    current["metabolites"][met.group(1)] = float(met.group(2))
            elif block == "subsystem":
                sub = sub_re.match(line)
                if sub:
                    current["subsystem"].append(_unquote(sub.group(1)))
    return reactions


def _read_tsv(path):
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            parts += [""] * (len(header) - len(parts))
            rows.append({h: _unquote(p) for h, p in zip(header, parts)})
    return rows


def _unquote(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text


# ------------------------------------------------------------------ mito gene sets

def mito_reaction_genes(reaction, resolver, drop_symbols=()):
    """
    Genes on a mitoMAMMALmod reaction, as MGI accessions, plus a flag for its rule state.

    The GPR is the primary source, but a few of its Ensembl IDs are retired and no
    longer resolve (Nnt is one), so the symbols in the GENE_LIST note are read as a
    second pass. Falling back keeps the audit conservative: a gene that either source
    can see is not reported as missing.
    """
    rule = reaction.get("gene_reaction_rule", "") or ""
    notes = reaction.get("notes") or {}
    stripped = rule.strip().lower()
    if not stripped:
        state = "empty"
    elif stripped in GPR_PLACEHOLDERS:
        state = GPR_PLACEHOLDERS[stripped]
    else:
        state = "genes"

    accessions = set()
    for ensembl in re.findall(r"ENSMUSG\d+", rule):
        acc = resolver.from_ensembl(ensembl)
        if acc:
            accessions.add(acc)
    for token in gpr_tokens(notes.get("GENE_LIST", "")):
        acc = resolver.from_symbol(token)
        if acc:
            accessions.add(acc)

    if drop_symbols:
        accessions = {a for a in accessions if resolver.symbol(a) not in drop_symbols}
    return accessions, state


# ----------------------------------------------------------------------- the audit

def audit(mito_model, ref_dir, drop_symbols=()):
    """
    Return one row per (mitoMAMMALmod reaction, candidate gene) pair.

    Reactions are matched to Mouse-GEM three ways -- shared KEGG reaction ID, shared
    BiGG/Recon3D ID, and identical metabolite sets after mapping metabolites through
    their KEGG/BiGG IDs. EC number is a fallback for reactions none of those three
    reach, because one EC spans every compartment variant and fans out badly on its
    own. Matches are restricted to Mouse-GEM reactions in compartments mitoMAMMAL models.
    """
    resolver = GeneResolver(ref_dir)
    mg_reactions, rxn_table, met_table = load_mouse_gem(ref_dir)

    # --- metabolite bridge
    base_of = {mid: row["metsNoComp"] for mid, row in met_table.items()}
    by_kegg, by_bigg = collections.defaultdict(set), collections.defaultdict(set)
    for mid, row in met_table.items():
        comp = mid[-1]
        if row["metKEGGID"]:
            by_kegg[(row["metKEGGID"], comp)].add((row["metsNoComp"], comp))
        for key in (row["metBiGGID"], row["metRecon3DID"]):
            if key:
                by_bigg[(key, comp)].add((row["metsNoComp"], comp))

    mito_met_options, mito_met_comp = {}, {}
    for met in mito_model["metabolites"]:
        comp = COMPARTMENTS[met["compartment"]]
        notes = met.get("notes") or {}
        mito_met_comp[met["id"]] = comp
        mito_met_options[met["id"]] = (
            set(by_kegg.get((notes.get("KEGG ID", ""), comp), set()))
            | by_bigg.get((notes.get("RECON2", ""), comp), set())
            | by_bigg.get((met["id"].rsplit("_", 1)[0], comp), set())
        )

    # --- index Mouse-GEM
    kegg_idx = collections.defaultdict(list)
    bigg_idx = collections.defaultdict(list)
    ec_idx = collections.defaultdict(list)
    sig_idx = collections.defaultdict(list)
    mg_by_id = {}
    gene_reaction_count = collections.Counter()
    gene_in_complex = collections.Counter()
    for rxn in mg_reactions:
        table = rxn_table[rxn["id"]]
        rxn["_ec"] = {e for e in re.split(r"[;, ]+", rxn.get("eccodes", ""))
                      if e.count(".") == 3 and not e.endswith(".-")}
        rxn["_sig"] = frozenset((base_of.get(m, m), m[-1]) for m in rxn["metabolites"])
        rxn["_comps"] = {m[-1] for m in rxn["metabolites"]}
        rxn["_genes"] = {a for a in (resolver.from_symbol(s)
                                     for s in gpr_tokens(rxn["gene_reaction_rule"])) if a}
        mg_by_id[rxn["id"]] = rxn
        for kegg in re.split(r"[;, ]+", table["rxnKEGGID"]):
            if kegg.startswith("R"):
                kegg_idx[kegg].append(rxn["id"])
        for bigg in (table["rxnBiGGID"], table["rxnRecon3DID"], table["rxnHepatoNET1ID"]):
            if bigg:
                bigg_idx[bigg].append(rxn["id"])
        for ec in rxn["_ec"]:
            ec_idx[ec].append(rxn["id"])
        sig_idx[rxn["_sig"]].append(rxn["id"])
        # keyed by accession, not symbol: Mouse-GEM still calls Coxfa4 'Ndufa4', and a
        # symbol-keyed lookup silently reports every renamed gene as a non-complex one
        for acc in rxn["_genes"]:
            gene_reaction_count[acc] += 1
            if " and " in rxn["gene_reaction_rule"]:
                gene_in_complex[acc] += 1

    symbols_in_mito = set()
    for rxn in mito_model["reactions"]:
        symbols_in_mito |= {resolver.symbol(a)
                            for a in mito_reaction_genes(rxn, resolver, drop_symbols)[0]}

    rows, matched, matched_strong = [], 0, 0
    for rxn in mito_model["reactions"]:
        notes = rxn.get("notes") or {}
        own_genes, gpr_state = mito_reaction_genes(rxn, resolver, drop_symbols)
        comps = {mito_met_comp.get(m, m.rsplit("_", 1)[-1]) for m in rxn["metabolites"]}
        allowed = set().union(*(COMPARTMENT_COMPATIBLE.get(c, {c}) for c in comps)) if comps else set()

        options = [mito_met_options.get(m, set()) for m in rxn["metabolites"]]
        union = set().union(*options) if options else set()

        hits = collections.defaultdict(set)
        for kegg in re.split(r"[;, ]+|\s+or\s+", notes.get("KEGG id", "")):
            if kegg.startswith("R"):
                for rid in kegg_idx.get(kegg, []):
                    hits[rid].add("kegg")
        for bigg in {rxn["id"], notes.get("Recon2 id", "")}:
            for rid in bigg_idx.get(bigg, []):
                hits[rid].add("bigg")
        if union and all(options):
            for sig, rids in sig_idx.items():
                if len(sig) == len(rxn["metabolites"]) and sig <= union:
                    for rid in rids:
                        hits[rid].add("struct")
        if hits:
            matched_strong += 1
        else:
            for ec in re.split(r"[;,]+|\s+or\s+|\s+and\s+", notes.get("EC Number", "").strip()):
                if ec.count(".") == 3 and not ec.endswith(".-"):
                    for rid in ec_idx.get(ec, []):
                        if mg_by_id[rid]["_comps"] <= allowed:
                            hits[rid].add("ec")
        if hits:
            matched += 1

        per_gene = collections.defaultdict(
            lambda: {"methods": set(), "rxns": [], "comps": set(), "shared": set(), "subs": set()})
        for rid, methods in hits.items():
            mg_rxn = mg_by_id[rid]
            if not mg_rxn["_comps"] <= allowed:
                continue
            shared = own_genes & mg_rxn["_genes"]
            for acc in mg_rxn["_genes"] - own_genes:
                entry = per_gene[acc]
                entry["methods"] |= methods
                entry["rxns"].append(rid)
                entry["comps"] |= mg_rxn["_comps"]
                entry["shared"] |= shared
                entry["subs"] |= set(mg_rxn["subsystem"])

        own_symbols = sorted(resolver.symbol(a) for a in own_genes)
        for acc, entry in per_gene.items():
            symbol = resolver.symbol(acc)
            strong = bool(entry["methods"] & {"kegg", "bigg", "struct"})
            family = sorted({s for s in own_symbols if _family(s) == _family(symbol)})
            if strong and entry["shared"]:
                tier = "A"
            elif strong:
                tier = "B"
            elif entry["shared"]:
                tier = "C"
            else:
                tier = "D"
            rows.append({
                "tier": tier,
                "candidate": symbol,
                "candidate_ensembl": resolver.acc_to_ens.get(acc, ""),
                "candidate_mgi": acc,
                "candidate_name": resolver.acc_to_name.get(acc, "")[:80],
                "mito_rxn": rxn["id"],
                "mito_subsystem": rxn.get("subsystem", ""),
                "mito_rxn_description": (notes.get("Description", rxn["name"]) or "").replace("&gt;", ">")[:90],
                "mito_ec": notes.get("EC Number", ""),
                "mito_gpr_state": gpr_state,
                "mito_gpr_genes": ",".join(own_symbols) or "(none)",
                "evidence": ",".join(sorted(entry["methods"])),
                "n_shared_genes": len(entry["shared"]),
                "shared_genes": ",".join(sorted(resolver.symbol(a) for a in entry["shared"])),
                "paralogue_of": ",".join(family),
                "already_elsewhere_in_model": symbol in symbols_in_mito,
                "mouse1_rxns": ",".join(sorted(entry["rxns"])[:8]),
                "n_mouse1_rxns": len(entry["rxns"]),
                "mouse1_compartments": "".join(sorted(entry["comps"])),
                "mouse1_subsystems": ";".join(sorted(entry["subs"]))[:80],
                "candidate_promiscuity": gene_reaction_count.get(acc, 0),
                "candidate_context": "complex" if gene_in_complex.get(acc, 0) else "isozyme",
            })

    stats = {
        "mito_reactions": len(mito_model["reactions"]),
        "matched": matched,
        "matched_strong": matched_strong,
        "mouse_gem_reactions": len(mg_reactions),
        "rows": len(rows),
        "candidate_genes": len({r["candidate"] for r in rows}),
    }
    rows.sort(key=lambda r: (r["tier"], r["mito_subsystem"], r["mito_rxn"], r["candidate"]))
    return rows, stats


def _family(symbol):
    """Crude gene-family root, so Ldha/Ldhb or Atp5f1a/Atp5f1b group together."""
    return re.sub(r"[0-9]+[a-z]*\d*$", "", symbol).lower()


def run(model_path="default", ref_dir=None, out_path="gene_audit.tsv",
        refresh=False, drop_symbols=()):
    ref_dir = ref_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "refs")
    print("=== mitoMAMMALmod GPR audit against Mouse-GEM ===")
    version = fetch_references(ref_dir, force=refresh)
    print(f"Mouse-GEM version {version}")
    with open(resolve_model_path(model_path), encoding="utf-8") as fh:
        mito_model = json.load(fh)
    rows, stats = audit(mito_model, ref_dir, drop_symbols=drop_symbols)
    print(f"matched {stats['matched']}/{stats['mito_reactions']} mitoMAMMALmod reactions "
          f"({stats['matched_strong']} on ID or stoichiometry, the rest on EC)")
    print(f"{stats['rows']} candidate rows covering {stats['candidate_genes']} distinct genes")
    by_tier = collections.Counter(r["tier"] for r in rows)
    for tier in "ABCD":
        genes = {r["candidate"] for r in rows if r["tier"] == tier}
        print(f"  tier {tier}: {by_tier[tier]:4d} rows, {len(genes):3d} genes")
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}")
    return rows, stats


def validate(model_path="default", ref_dir=None):
    """Delete the three genes already known to have been missing, and check they come back."""
    controls = ("Ldha", "Pfkl", "Pfkp")
    print(f"=== positive control: removing {', '.join(controls)} from the model's GPRs ===")
    rows, _ = run(model_path=model_path, ref_dir=ref_dir, out_path=None, drop_symbols=controls)
    ok = True
    for gene in controls:
        found = [r for r in rows if r["candidate"] == gene]
        top = min((r["tier"] for r in found), default="-")
        print(f"  {gene}: {len(found)} row(s), best tier {top}, "
              f"on {sorted({r['mito_rxn'] for r in found})}")
        ok &= bool(found) and top == "A"
    print("PASS" if ok else "FAIL")
    return ok


#: public name for the audit entry point
audit_model_genes = run


def main():
    parser = argparse.ArgumentParser(description="Audit mitoMAMMALmod GPRs against Mouse-GEM")
    parser.add_argument("--model", default="default", help="path to the mitoMAMMAL json")
    parser.add_argument("--refs", default=None, help="directory for the downloaded reference files")
    parser.add_argument("--out", default="gene_audit.tsv")
    parser.add_argument("--refresh", action="store_true", help="re-download the reference files")
    parser.add_argument("--validate", action="store_true", help="run the Ldha/Pfkl/Pfkp control only")
    args = parser.parse_args()
    if args.validate:
        validate(args.model, args.refs)
    else:
        run(args.model, args.refs, args.out, args.refresh)


if __name__ == "__main__":
    main()
