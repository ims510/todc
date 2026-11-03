"""Unit test: column merge using the user's actual feature-name convention,
including subtypes (rel_shallow=acl:relcl style).
"""
import sys
import unittest.mock as mock
import numpy as np

sys.modules["tod"] = mock.MagicMock()
sys.modules["tod.corpus"] = mock.MagicMock()

import importlib.util
spec = importlib.util.spec_from_file_location(
    "v", "/home/claude/treebank_analysis/03b_validate_candidates.py"
)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


# Subset of the user's real feature names — mixes bare relations, subtyped
# relations, and other (non-rel_shallow) features.
feature_names_raw = [
    "node:X:child:rel_shallow=acl",
    "node:X:child:rel_shallow=acl:relcl",
    "node:X:child:rel_shallow=advcl",
    "node:X:child:rel_shallow=advcl:cleft",
    "node:X:child:rel_shallow=aux",
    "node:X:child:rel_shallow=aux:caus",
    "node:X:child:rel_shallow=aux:pass",
    "node:X:child:rel_shallow=csubj",
    "node:X:child:rel_shallow=csubj:pass",
    "node:X:child:rel_shallow=nmod",
    "node:X:child:rel_shallow=nsubj",
    "node:X:child:rel_shallow=nsubj:pass",
    "node:X:child:rel_shallow=obj",
    "node:X:child:rel_shallow=obl",
    "node:X:child:rel_shallow=obl:agent",
    "node:X:child:rel_shallow=obl:arg",
    "node:X:child:rel_shallow=obl:mod",
    "node:X:child:upos=NOUN",
    "node:X:child:upos=VERB",
]
idx2feature = dict(enumerate(feature_names_raw))
inv = {v: k for k, v in idx2feature.items()}
n_cols = len(idx2feature)

M = np.zeros((4, n_cols))
# verb_1
M[0, inv["node:X:child:rel_shallow=obl"]]         = 0.40
M[0, inv["node:X:child:rel_shallow=obl:agent"]]   = 0.05
M[0, inv["node:X:child:rel_shallow=nsubj"]]       = 0.50
M[0, inv["node:X:child:rel_shallow=obj"]]         = 0.30
M[0, inv["node:X:child:rel_shallow=aux"]]         = 0.20
M[0, inv["node:X:child:upos=NOUN"]]               = 0.80
# verb_2: passive
M[1, inv["node:X:child:rel_shallow=obl"]]         = 0.20
M[1, inv["node:X:child:rel_shallow=obl:agent"]]   = 0.30
M[1, inv["node:X:child:rel_shallow=aux:pass"]]    = 0.40
M[1, inv["node:X:child:rel_shallow=nsubj:pass"]]  = 0.50
M[1, inv["node:X:child:upos=NOUN"]]               = 0.70
# noun_1
M[2, inv["node:X:child:rel_shallow=nmod"]]        = 0.45
M[2, inv["node:X:child:rel_shallow=acl:relcl"]]   = 0.20
M[2, inv["node:X:child:rel_shallow=acl"]]         = 0.05
M[2, inv["node:X:child:upos=NOUN"]]               = 0.30
# noun_2
M[3, inv["node:X:child:rel_shallow=nmod"]]        = 0.60
M[3, inv["node:X:child:upos=NOUN"]]               = 0.20

orig_total = M.sum()
print(f"Total mass before: {orig_total:.4f}, cols: {n_cols}")

print("\n=== Test 1: obl -> nmod ===")
M_new, new_i2f, n_merged, n_new = m.apply_column_merge(M, idx2feature, {"obl": "nmod"})
print(f"  merges={n_merged}, new_cols={n_new}, total_cols={M_new.shape[1]}")
assert n_merged == 4, f"expected 4 merges (obl + 3 subtypes), got {n_merged}"
assert n_new == 3, f"expected 3 new cols (nmod:agent/arg/mod), got {n_new}"
assert np.isclose(M_new.sum(), orig_total), "mass not conserved"
# Source cols zeroed
for src in ["obl", "obl:agent", "obl:arg", "obl:mod"]:
    if f"node:X:child:rel_shallow={src}" in inv:
        i = inv[f"node:X:child:rel_shallow={src}"]
        assert np.allclose(M_new[:, i], 0), f"src {src} should be zeroed"
# bare nmod absorbed bare obl
i_nmod, i_obl = inv["node:X:child:rel_shallow=nmod"], inv["node:X:child:rel_shallow=obl"]
assert np.allclose(M_new[:, i_nmod], M[:, i_nmod] + M[:, i_obl])
# new nmod:agent column has the obl:agent value
inv_new = {v: k for k, v in new_i2f.items()}
i_nmod_agent = inv_new["node:X:child:rel_shallow=nmod:agent"]
i_obl_agent  = inv["node:X:child:rel_shallow=obl:agent"]
assert np.allclose(M_new[:, i_nmod_agent], M[:, i_obl_agent])
print("  OK")

print("\n=== Test 2: csubj -> nsubj (subtypes match existing target subtypes) ===")
M_new2, _, n_merged2, n_new2 = m.apply_column_merge(M, idx2feature, {"csubj": "nsubj"})
print(f"  merges={n_merged2}, new_cols={n_new2}")
assert n_merged2 == 2, f"expected 2 merges, got {n_merged2}"
assert n_new2 == 0, f"expected 0 new cols, got {n_new2}"
i_ns_pass = inv["node:X:child:rel_shallow=nsubj:pass"]
i_cs_pass = inv["node:X:child:rel_shallow=csubj:pass"]
assert np.allclose(M_new2[:, i_ns_pass], M[:, i_ns_pass] + M[:, i_cs_pass])
print("  OK")

print("\n=== Test 3: parse_rel_feature ===")
cases = [
    ("node:X:child:rel_shallow=acl",        "acl",  ""),
    ("node:X:child:rel_shallow=acl:relcl",  "acl",  ":relcl"),
    ("node:X:child:rel_shallow=obl:agent",  "obl",  ":agent"),
    ("node:X:child:upos=NOUN",              None,   None),
]
for name, exp_base, exp_sub in cases:
    parsed = m.parse_rel_feature(name)
    if exp_base is None:
        assert parsed is None, f"{name}: should not parse, got {parsed}"
        print(f"  OK {name!r} -> None")
    else:
        prefix, base, subtype, tail = parsed
        assert base == exp_base and subtype == exp_sub
        print(f"  OK {name!r} -> base={base!r} subtype={subtype!r}")

print("\n=== Test 4: list_base_relations ===")
bases = m.list_base_relations(idx2feature)
print(f"  bases: {bases}")
assert set(bases) == {"acl", "advcl", "aux", "csubj", "nmod", "nsubj", "obj", "obl"}
print("  OK")

print("\nAll tests passed!")
