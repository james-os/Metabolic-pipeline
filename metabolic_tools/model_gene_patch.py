"""
Turn the Mouse-GEM audit into a reviewable decision table, and apply it to the model.

Two steps, deliberately separate so the curation call stays visible and editable:

    build_decisions()  audit TSV  -> decisions TSV with a recommend/reason column
    apply_decisions()  decisions TSV -> patched mitoMAMMAL json

The filters are all tissue-independent, so a patched model stays reusable blind:

  * boundary and transport reactions are left alone
  * MitoCarta3.0 localisation, used asymmetrically -- presence is positive evidence a
    gene is mitochondrial, so a MitoCarta gene is refused on a cytosol-only reaction;
    absence is weak evidence, so it only refuses a gene on a mitochondria-only
    reaction when the genes already curated onto that reaction are MitoCarta members
  * an explicit list of Mouse-GEM GPR artefacts -- genes with no metabolic role that
    Mouse1 carries on metabolic reactions (Taf9 on adenylate kinase, Vps29 on
    phosphoserine phosphatase, and so on)

Everything else is included, on the argument that an isozyme which is not expressed
cannot move an OR-rule score, and that a published model should not be tuned to one
tissue. Note that argument holds exactly under or_strategy='max'; under the
ecs_calculator default of 'sum', extra isozymes do accumulate ambient signal, which
is what the variant comparison is meant to measure.

Usage
-----
    python -m metabolic_tools.model_gene_patch --decisions decisions.tsv
    python -m metabolic_tools.model_gene_patch --apply decisions.tsv --variant inclusive \\
        --out metabolic_tools/data/mitoMAMMALmod_inclusive.json
"""

import argparse
import ast
import collections
import csv
import json
import os
import re

from .gene_mapping import resolve_model_path

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MITOCARTA_TSV = os.path.join(DATA_DIR, "mitocarta3_mouse.tsv")

# Genes Mouse-GEM carries in GPRs for reactions they have no metabolic role in.
# Read-through loci, overlapping transcripts and a handful of plain annotation slips.
MOUSE1_GPR_ARTEFACTS = {
    "Atg5lrt",    # on ATP synthase
    "Cpne7",      # on ALAS
    "Crisp1",     # on phosphoglycerate kinase
    "Olah",       # on fatty acid synthase
    "Rwdd2a",     # on malic enzyme
    "Sccpdh",     # on saccharopine dehydrogenase
    "Taf9",       # on adenylate kinase
    "Tcf4",       # on malic enzyme
    "Tmem91",     # on the branched-chain ketoacid dehydrogenase reactions
    "Tnrc6b",     # on adenylosuccinate lyase
    "Ubac2",      # on phosphoglycerate dehydrogenase
    "Vps29",      # on phosphoserine phosphatase
}

# The variants the comparison builds. 'repaired' adds no genes and serves as the
# control; 'conservative' keeps only gaps that are hard to read as deliberate;
# 'inclusive' is everything the filters above allow.
VARIANTS = {
    "repaired": set(),
    "conservative": {"A"},
    "inclusive": {"A", "B"},
}


def split_top_level_or(rule):
    """Split a GPR on its top-level 'or', leaving bracketed sub-rules intact."""
    parts, depth, current = [], 0, []
    tokens = re.split(r"(\(|\)|\bor\b|\band\b)", rule)
    for token in tokens:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif token == "or" and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(token)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _is_mouse_branch(text, symbols=False):
    """Mouse branches carry ENSMUSG ids, or MGI-cased symbols rather than human ones."""
    if not symbols:
        return "ENSMUSG" in text
    return any(re.match(r"^(mt-)?[A-Z][a-z0-9]", t)
               for t in re.findall(r"[A-Za-z0-9_\-\.]+", text) if t not in ("and", "or"))


# Defects in the shipped mitoMAMMALmod that break the GPRs downstream. Each is a
# literal substitution, applied to the field named, so the change stays auditable.
# Only the first two and the Ttc19 typo affect mouse scores; the rest are tidying.
MODEL_REPAIRS = {
    # Unclosed bracket, so GENE_LIST cannot be split into its mouse and human branches.
    ("CV_mitoMap", "GENE_LIST"): [
        ("(Atp5g1 or Atp5g2 or Atp5g3) or (ATP5A1",
         "(Atp5g1 or Atp5g2 or Atp5g3)) or (ATP5A1"),
        ("and MT-ATP8) and (ATP5G1 or ATP5G2 or ATP5G3)",
         "and MT-ATP8 and (ATP5G1 or ATP5G2 or ATP5G3))"),
    ],
    ("ACACt2mB_mitoMap", "GENE_LIST"): [("(MPC1 and MPC2", "(MPC1 and MPC2)")],
    # Ttc19 carries an extra digit, so it never matches the data. It sits in an AND
    # chain, which drags the whole of complex III to the floor under min/median-AND.
    ("CIII_mitoMap", "GENE_ASSOCIATION"): [("ENSMUSG000000422989", "ENSMUSG00000042298")],
    # Nnt is the one gene whose id moved: the Kolla E16 matrix is annotated with the
    # retired ENSMUSG00000116207, newer references use ENSMUSG00000025453. Swapping it
    # breaks the older data, so the rule carries both and matches whichever is present.
    ("NNT_mitoMap", "gene_reaction_rule"): [
        ("ENSMUSG00000116207", "(ENSMUSG00000116207 or ENSMUSG00000025453)")],
    ("NNT_mitoMap", "GENE_ASSOCIATION"): [
        ("ENSMUSG00000116207", "(ENSMUSG00000116207 or ENSMUSG00000025453)")],
    ("NNT_mitoMap", "GENE_LIST"): [("Nnt or NNT", "(Nnt or Nnt) or NNT")],
    # Human ids of the wrong length; harmless for mouse runs, wrong either way.
    ("CI_mitoMap", "gene_reaction_rule"): [("ENSG000001152863", "ENSG00000115286")],
    ("GLYOXm", "gene_reaction_rule"): [("ENSG0000006385", "ENSG00000063854")],
}


def _apply_literal_repairs(reaction):
    """Apply the MODEL_REPAIRS substitutions for one reaction, returning how many fired."""
    fired = 0
    notes = reaction.setdefault("notes", {})
    for (rxn_id, field), subs in MODEL_REPAIRS.items():
        if rxn_id != reaction["id"]:
            continue
        target = reaction if field == "gene_reaction_rule" else notes
        text = target.get(field, "")
        for old, new in subs:
            if old in text:
                text = text.replace(old, new)
                fired += 1
        target[field] = text
    return fired


def _repair_gene_association(reaction):
    """
    mitoMAMMALmod ships two reactions whose GENE_ASSOCIATION note is malformed -- an
    embedded newline on r1451 and an unclosed bracket on OIVD2m. ecs_calculator reads
    that note in preference to gene_reaction_rule and returns zeros when it will not
    parse, so both reactions currently score zero for every cell. Where the note is
    broken and the rule is not, the rule wins.
    """
    notes = reaction.get("notes") or {}
    assoc = notes.get("GENE_ASSOCIATION")
    rule = reaction.get("gene_reaction_rule", "") or ""
    if assoc is None or not _parses(rule):
        return False
    # GluForTx keeps only the human ortholog in the note, so the reaction scores zero
    # on mouse data even though its rule names Ftcd. Any note that has lost mouse genes
    # the rule still carries is treated the same way.
    lost_mouse = set(re.findall(r"ENSMUSG\d+", rule)) - set(re.findall(r"ENSMUSG\d+", assoc))
    if _parses(assoc) and not lost_mouse:
        return False
    notes["GENE_ASSOCIATION"] = rule
    return True


def _parses(rule):
    rule = (rule or "").strip()
    if not rule:
        return True
    safe = re.sub(r"(?<![A-Za-z0-9_\-\.])(?!and\b|or\b)[A-Za-z0-9_\-\.]+", "V", rule)
    try:
        ast.parse(safe, mode="eval")
        return True
    except SyntaxError:
        return False


def load_mitocarta(path=MITOCARTA_TSV):
    """Symbol -> sub-mitochondrial localisation, from the packaged MitoCarta3.0 extract."""
    table = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            table[row["symbol"]] = row["sub_localization"] or "unspecified"
    return table


def _reaction_compartments(mito_model):
    """Reaction id -> the set of compartments its metabolites sit in."""
    comp_of = {m["id"]: m["compartment"] for m in mito_model["metabolites"]}
    short = {"Cytosol": "c", "Mitochondrion": "m", "external": "e"}
    return {r["id"]: {short.get(comp_of.get(m, ""), "?") for m in r["metabolites"]}
            for r in mito_model["reactions"]}


def build_decisions(audit_path, model_path="default", out_path="decisions.tsv"):
    """Add recommend/reason columns to the audit, and a class for each candidate."""
    with open(resolve_model_path(model_path), encoding="utf-8") as fh:
        mito_model = json.load(fh)
    comps = _reaction_compartments(mito_model)
    mitocarta = load_mitocarta()

    with open(audit_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    # which reactions mitoMAMMAL itself models as a complex
    is_complex = {r["id"]: " and " in ((r.get("notes") or {}).get("GENE_ASSOCIATION", "") or "")
                  for r in mito_model["reactions"]}

    for row in rows:
        rxn = row["mito_rxn"]
        candidate = row["candidate"]
        rxn_comps = comps.get(rxn, set())
        existing = [g for g in row["mito_gpr_genes"].split(",") if g and g != "(none)"]

        row["rxn_compartments"] = "".join(sorted(rxn_comps))
        row["candidate_mitocarta"] = mitocarta.get(candidate, "")
        row["existing_mitocarta"] = ",".join(
            f"{g}:{mitocarta[g]}" for g in existing if g in mitocarta)
        row["candidate_class"] = (
            "complex_subunit" if is_complex.get(rxn) and row["candidate_context"] == "complex"
            else "isozyme" if row["paralogue_of"] else "other_enzyme")

        boundary = (row["mito_subsystem"].lower().startswith("boundary")
                    or row["mito_gpr_state"] in ("empty", "unknown", "non_enzymatic"))
        existing_is_mito = any(g in mitocarta for g in existing)

        if candidate in MOUSE1_GPR_ARTEFACTS:
            recommend, reason = "exclude", "mouse1_gpr_artefact"
        elif boundary:
            recommend, reason = "exclude", "boundary_or_ungened_reaction"
        elif rxn_comps == {"c"} and candidate in mitocarta and not existing_is_mito:
            # Only a gap when mitoMAMMAL's own genes for the reaction are cytosolic too.
            # Reactions on the outer membrane (CPT1) are modelled as cytosolic but
            # already carry MitoCarta genes, and must not be filtered on that basis.
            recommend, reason = "exclude", "mitochondrial_gene_on_cytosolic_reaction"
        elif rxn_comps <= {"m"} and candidate not in mitocarta and existing_is_mito:
            recommend, reason = "exclude", "non_mitochondrial_gene_on_mitochondrial_reaction"
        elif row["tier"] in ("C", "D"):
            recommend, reason = "exclude", "ec_only_match"
        else:
            recommend, reason = "include", f"{row['candidate_class']}_tier_{row['tier']}"

        row["recommend"] = recommend
        row["reason"] = reason

    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    kept = [r for r in rows if r["recommend"] == "include"]
    print(f"wrote {out_path}")
    print(f"  include: {len(kept)} rows, {len({r['candidate'] for r in kept})} genes, "
          f"{len({r['mito_rxn'] for r in kept})} reactions")
    for reason, n in collections.Counter(r["reason"] for r in rows).most_common():
        print(f"    {reason:52s} {n:4d}")
    return rows


def apply_decisions(decisions_path, variant="inclusive", model_path="default", out_path=None):
    """Write a copy of the model with the accepted genes added to their GPRs."""
    tiers = VARIANTS[variant]
    with open(decisions_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["recommend"] == "include" and r["tier"] in tiers]
    if variant == "conservative":
        rows = [r for r in rows if r["candidate_class"] == "complex_subunit"
                or r["candidate_mitocarta"] or r["already_elsewhere_in_model"] == "True"]

    with open(resolve_model_path(model_path), encoding="utf-8") as fh:
        model = json.load(fh)

    additions = collections.defaultdict(list)
    for row in rows:
        additions[row["mito_rxn"]].append(row)

    declared = {g["id"] for g in model["genes"]}
    n_rxn = n_gene = n_repaired = 0
    for rxn in model["reactions"]:
        n_repaired += _apply_literal_repairs(rxn)
        if _repair_gene_association(rxn):
            n_repaired += 1
        new = additions.get(rxn["id"])
        if not new:
            continue
        notes = rxn.setdefault("notes", {})
        rule = rxn.get("gene_reaction_rule", "") or ""
        assoc = notes.get("GENE_ASSOCIATION", rule) or rule
        genes = notes.get("GENE_LIST", "")

        for row in sorted(new, key=lambda r: r["candidate"]):
            symbol, ensembl = row["candidate"], row["candidate_ensembl"]
            if not ensembl:
                continue
            existing = [g for g in row["mito_gpr_genes"].split(",") if g and g != "(none)"]
            # A subunit suffixed onto an existing one (Ndufb4 -> Ndufb4b) is a paralogue
            # of that subunit and belongs beside it; anything else the complex needs is
            # a subunit in its own right and has to be ANDed in.
            sibling = next((g for g in existing
                            if symbol.startswith(g) and len(symbol) - len(g) <= 2), None)
            if row["candidate_class"] == "complex_subunit" and sibling:
                mode, anchor = "or_beside", sibling
            elif row["candidate_class"] == "complex_subunit":
                mode, anchor = "and_into_branch", None
            else:
                mode, anchor = "or_top_level", None

            anchor_id = _ensembl_of(model, anchor) if anchor else None
            rule = _insert(rule, ensembl, mode, anchor_id, symbols=False)
            assoc = _insert(assoc, ensembl, mode, anchor_id, symbols=False)
            genes = _insert(genes, symbol, mode, anchor, symbols=True)

            if ensembl not in declared:
                model["genes"].append({"id": ensembl, "name": symbol})
                declared.add(ensembl)
                n_gene += 1
        rxn["gene_reaction_rule"] = rule
        notes["GENE_ASSOCIATION"] = assoc
        notes["GENE_LIST"] = genes
        n_rxn += 1

    model["id"] = f"{model.get('id', 'mitoMAP')}_{variant}"
    model["name"] = f"{model.get('name', 'mitoMAP')} ({variant} Mouse-GEM GPR patch)"

    n_declared = _reconcile_gene_list(model)
    problems = validate_model(model, variant)
    out_path = out_path or os.path.join(DATA_DIR, f"mitoMAMMALmod_{variant}.json")
    if problems:
        raise ValueError(f"{variant}: {len(problems)} GPR problem(s); model not written")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(model, fh)
    print(f"{variant}: patched {n_rxn} reactions, added {len(rows)} gene assignments "
          f"({n_gene} genes new to the model), {n_repaired} repairs, {n_declared} genes newly declared")
    print(f"wrote {out_path}")
    return out_path


def _reconcile_gene_list(model):
    """
    Declare every gene the GPRs actually use.

    The three genes added to mitoMAMMALmod by hand (Ldha, Pfkl, Pfkp) went into the
    rules but never into model['genes'], so cobra loads them as auto-created objects
    and anything reading the gene list alone misses them.
    """
    declared = {g["id"] for g in model["genes"]}
    symbols = {}
    for rxn in model["reactions"]:
        notes = rxn.get("notes") or {}
        ids = re.findall(r"ENS[A-Z]*G\d+", rxn.get("gene_reaction_rule", "") or "")
        names = [t for t in re.findall(r"[A-Za-z0-9_\-\.]+", notes.get("GENE_LIST", ""))
                 if t not in ("and", "or")]
        if len(ids) == len(names):
            symbols.update(dict(zip(ids, names)))
    added = 0
    for rxn in model["reactions"]:
        for gene in re.findall(r"ENS[A-Z]*G\d+", rxn.get("gene_reaction_rule", "") or ""):
            if gene not in declared:
                model["genes"].append({"id": gene, "name": symbols.get(gene, gene)})
                declared.add(gene)
                added += 1
    return added


def validate_model(model, label=""):
    """
    Refuse to write a model whose GPRs would misbehave downstream.

    evaluate_gpr_node returns zeros for a rule it cannot parse, so a bad bracket is
    silent at runtime and shows up only as a reaction that scores zero everywhere.
    """
    problems = []
    declared = {g["id"] for g in model["genes"]}
    for rxn in model["reactions"]:
        notes = rxn.get("notes") or {}
        rule = rxn.get("gene_reaction_rule", "") or ""
        assoc = notes.get("GENE_ASSOCIATION", rule) or rule
        for field, text in (("gene_reaction_rule", rule), ("GENE_ASSOCIATION", assoc),
                            ("GENE_LIST", notes.get("GENE_LIST", ""))):
            if text and not _balanced(text):
                problems.append(f"{rxn['id']}: unbalanced brackets in {field}")
            if field != "GENE_LIST" and not _parses(text):
                problems.append(f"{rxn['id']}: {field} does not parse")
        for gene in re.findall(r"ENS[A-Z]*G\d+", rule):
            if gene not in declared:
                problems.append(f"{rxn['id']}: {gene} missing from the model gene list")
        # the mouse genes in the rule and in the note have to stay in step
        if set(re.findall(r"ENSMUSG\d+", rule)) != set(re.findall(r"ENSMUSG\d+", assoc)):
            problems.append(f"{rxn['id']}: gene_reaction_rule and GENE_ASSOCIATION disagree")
    if problems:
        print(f"  {label} validation found {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    {problem}")
    return problems


def _balanced(text):
    depth = 0
    for char in text or "":
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    return depth == 0


def _ensembl_of(model, symbol):
    """The model's Ensembl id for a mouse symbol, read back out of the GENE_LIST notes."""
    for rxn in model["reactions"]:
        notes = rxn.get("notes") or {}
        ids = re.findall(r"ENS[A-Z]*G\d+", rxn.get("gene_reaction_rule", "") or "")
        names = [t for t in re.findall(r"[A-Za-z0-9_\-\.]+", notes.get("GENE_LIST", ""))
                 if t not in ("and", "or")]
        if len(ids) == len(names) and symbol in names:
            return ids[names.index(symbol)]
    return None


def _insert(text, addition, mode, anchor, symbols):
    """Place one gene into a GPR: beside an existing one, into the species branch, or on top."""
    text = (text or "").strip()
    if not text:
        return addition
    if mode == "or_beside" and anchor:
        pattern = rf"(?<![A-Za-z0-9_\-\.]){re.escape(anchor)}(?![A-Za-z0-9_\-\.])"
        if re.search(pattern, text):
            return re.sub(pattern, f"({anchor} or {addition})", text, count=1)
    if mode == "and_into_branch":
        branches = split_top_level_or(text)
        if any(_is_mouse_branch(b, symbols) for b in branches):
            rebuilt = []
            for branch in branches:
                if _is_mouse_branch(branch, symbols):
                    inner = branch.strip()
                    if inner.startswith("(") and inner.endswith(")"):
                        inner = inner[1:-1].strip()
                    rebuilt.append(f"({inner} and {addition})")
                else:
                    rebuilt.append(branch)
            return " or ".join(rebuilt)
    if " and " in text.lower() or " or " in text.lower():
        text = f"({text})"
    return f"{text} or {addition}"


def main():
    parser = argparse.ArgumentParser(description="Build and apply mitoMAMMALmod GPR patches")
    parser.add_argument("--audit", default=os.path.join(DATA_DIR, "mitoMAMMALmod_gene_audit.tsv"))
    parser.add_argument("--decisions", default=None, help="write the decision table here")
    parser.add_argument("--apply", default=None, help="decision table to apply")
    parser.add_argument("--variant", default="inclusive", choices=sorted(VARIANTS))
    parser.add_argument("--model", default="default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.decisions:
        build_decisions(args.audit, args.model, args.decisions)
    if args.apply:
        apply_decisions(args.apply, args.variant, args.model, args.out)


if __name__ == "__main__":
    main()
