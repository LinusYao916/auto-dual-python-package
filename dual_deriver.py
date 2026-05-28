"""
dual_deriver.py
================

Automated Lagrangian / Dual-LP derivation tool built on SymPy.

DESIGN: I/O FORMAT
------------------
A primal LP is built declaratively against an instance of ``DualDeriver``.

    deriver = DualDeriver(sense='min')

    # Bare SymPy symbols are used for indices,
    # IndexedBase for variables and parameters.
    i, j, k = sp.symbols('i j k')
    c, a, U, L, x, d, e = (sp.IndexedBase(s) for s in 'c a U L x d e'.split())

    y    = deriver.declare_var('y',   index_sets=['J','K_j'],
                               free_symbols=[j,k], lower=0)
    phi  = deriver.declare_var('phi', index_sets=['I'],
                               free_symbols=[i],   lower=0)

    deriver.set_objective(
        SUM(c[j,k]*y[j,k], (j,'J'), (k,'K_j'))     # nested-summation term
      + SUM(e[i]*phi[i],   (i,'I'))
    )

    deriver.add_constraint(
        lhs   = SUM(a[i,j,k]*y[j,k], (j,'J'), (k,'K_j')) + TERM(phi[i]),
        sense = '>=',
        rhs   = TERM(d[i]),
        forall      = [i],   forall_sets = ['I'],
        multiplier_name = 'lambda',
    )
    ...
    result = deriver.derive_dual()
    print(render_dual(result))

Two tiny constructors keep the syntax tight:
  *  ``SUM(body, (dummy, set_name), ...)`` --  Σ_{dummy ∈ set_name} body
  *  ``TERM(body)``                        --  body  (no summation)

They both return a ``SymExpr`` (a list of ``SumTerm``s), and SymExprs support
``+``, ``-`` and ``-self`` so building an algebraic primal is natural.

The same constructors are used in the validation test to build the
*expected* dual, and we compare the derived vs expected SymExprs in a
structure-aware way (a canonical (body, dummies, sets) → coefficient map).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set

import sympy as sp
from sympy import (
    IndexedBase, Indexed, Symbol, S, latex, Add, Mul,
)


# =============================================================================
# 1.  Internal symbolic algebra:  SumTerm  and  SymExpr
# =============================================================================
#
# A SumTerm represents a *single* summed monomial
#
#         sign · Σ_{d_1 ∈ S_1, ..., d_n ∈ S_n}  body
#
# `body` is an ordinary SymPy expression that may use both the dummies and
# any *free* outer index symbols.  We keep the summations symbolic (we never
# evaluate them) so the printed LaTeX comes out the way an OR person would
# write it.  A SymExpr is just a list of SumTerms.
# -----------------------------------------------------------------------------


@dataclass
class SumTerm:
    body: sp.Expr
    dummies: Tuple[Symbol, ...] = ()
    sets:    Tuple[str, ...]    = ()
    sign:    int                = 1

    # -- mutation helpers ----------------------------------------------------
    def negated(self) -> "SumTerm":
        return SumTerm(self.body, self.dummies, self.sets, -self.sign)

    def with_outer_sum(self, idx: Symbol, set_name: str) -> "SumTerm":
        return SumTerm(self.body,
                       (idx,) + self.dummies,
                       (set_name,) + self.sets,
                       self.sign)

    def multiplied_by(self, factor: sp.Expr) -> "SumTerm":
        return SumTerm(factor * self.body, self.dummies, self.sets, self.sign)

    def split_addends(self) -> List["SumTerm"]:
        """Expand and split the body so each SumTerm holds one monomial only."""
        body = sp.expand(self.body)
        if body == 0:
            return []
        if isinstance(body, Add):
            out: List[SumTerm] = []
            for a in body.args:
                if a.could_extract_minus_sign():
                    out.append(SumTerm(-a, self.dummies, self.sets, -self.sign))
                else:
                    out.append(SumTerm(a, self.dummies, self.sets, self.sign))
            return out
        if body.could_extract_minus_sign():
            return [SumTerm(-body, self.dummies, self.sets, -self.sign)]
        return [SumTerm(body, self.dummies, self.sets, self.sign)]

    # -- display -------------------------------------------------------------
    def to_latex(self) -> str:
        sigil = "+" if self.sign > 0 else "-"
        sums = "".join(rf"\sum_{{{latex(d)} \in {latex(Symbol(s))}}} "
                       for d, s in zip(self.dummies, self.sets))
        return f"{sigil} {sums}{latex(self.body)}"


class SymExpr:
    """An additive combination of SumTerms."""

    def __init__(self, terms: Optional[Iterable[SumTerm]] = None):
        self.terms: List[SumTerm] = []
        for t in (terms or []):
            self.terms.extend(t.split_addends())

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other: "SymExpr") -> "SymExpr":
        return SymExpr(self.terms + other.terms)

    def __sub__(self, other: "SymExpr") -> "SymExpr":
        return SymExpr(self.terms + [t.negated() for t in other.terms])

    def __neg__(self) -> "SymExpr":
        return SymExpr([t.negated() for t in self.terms])

    def wrap_outer_sum(self, idx: Symbol, set_name: str) -> "SymExpr":
        return SymExpr([t.with_outer_sum(idx, set_name) for t in self.terms])

    def multiplied_by(self, factor: sp.Expr) -> "SymExpr":
        return SymExpr([t.multiplied_by(factor) for t in self.terms])

    # -- analysis ------------------------------------------------------------
    def diff_indexed(self, base: IndexedBase,
                     free: Tuple[Symbol, ...]) -> "SymExpr":
        """
        Symbolic partial derivative wrt the *indexed* variable ``base[free]``.

        Because the model is affine in the decision variables, this is simply
        the linear coefficient of ``base[free]``.  For a SumTerm whose body
        contains ``base[d]`` with `d` a dummy, we α-rename `d → free` so the
        coefficient is well-defined and the remaining dummies stay summed.
        """
        out: List[SumTerm] = []
        target = base[free] if len(free) > 1 else base[free[0]]

        for term in self.terms:
            for occ in term.body.atoms(Indexed):
                if occ.base != base:
                    continue
                occ_idx = tuple(occ.indices)
                if len(occ_idx) != len(free):
                    continue

                # Build the α-renaming   dummy_in_body  →  free_index
                subs_map: Dict[Symbol, Symbol] = {}
                ok = True
                for oi, fi in zip(occ_idx, free):
                    if oi in term.dummies:
                        if subs_map.get(oi, fi) != fi:
                            ok = False; break
                        subs_map[oi] = fi
                    elif oi == fi:
                        continue
                    else:
                        # Indices can't be matched, skip this occurrence.
                        ok = False; break
                if not ok:
                    continue

                renamed_body = term.body.subs(subs_map)
                coeff = renamed_body.coeff(target)            # affine ⇒ linear

                # Dummies that survive (those not absorbed by the renaming)
                rem_dummies = tuple(d for d in term.dummies
                                    if d not in subs_map)
                rem_sets    = tuple(s for d, s in zip(term.dummies, term.sets)
                                    if d not in subs_map)

                out.append(SumTerm(coeff, rem_dummies, rem_sets, term.sign))
        return SymExpr(out)

    def parts_without(self, var_bases: Set[IndexedBase]) -> "SymExpr":
        """Keep only the SumTerms whose body uses none of ``var_bases``."""
        keep = []
        for t in self.terms:
            bases = {a.base for a in t.body.atoms(Indexed)}
            if bases.isdisjoint(var_bases):
                keep.append(t)
        return SymExpr(keep)

    # -- comparison / display ------------------------------------------------
    def canonical(self) -> Dict[Tuple, int]:
        """
        Map (canonical-body, dummies, sets) → summed-sign.

        Two SymExprs are mathematically equal (as nested affine sums) iff
        their canonical maps agree.  This is what we use for ground-truth
        verification in the test.
        """
        out: Dict[Tuple, int] = {}
        for t in self.terms:
            key = (sp.expand(t.body), t.dummies, t.sets)
            out[key] = out.get(key, 0) + t.sign
        return {k: v for k, v in out.items() if v != 0}

    def to_latex(self) -> str:
        if not self.terms:
            return "0"
        s = " ".join(t.to_latex() for t in self.terms).strip()
        return s[1:].lstrip() if s.startswith("+") else s


def SUM(body: sp.Expr, *index_set_pairs) -> SymExpr:
    """SUM(c[j,k]*y[j,k], (j,'J'), (k,'K_j'))  →  Σ_{j∈J} Σ_{k∈K_j} c_{jk} y_{jk}"""
    dummies = tuple(p[0] for p in index_set_pairs)
    sets    = tuple(p[1] for p in index_set_pairs)
    return SymExpr([SumTerm(body, dummies, sets)])


def TERM(body: sp.Expr) -> SymExpr:
    """Non-summed term -- parameterised only by the surrounding free indices."""
    return SymExpr([SumTerm(body)])


# =============================================================================
# 2.  Model declaration containers
# =============================================================================


@dataclass
class VarInfo:
    base: IndexedBase
    index_sets: Tuple[str, ...]
    free_symbols: Tuple[Symbol, ...]
    lower: Optional[float] = 0          # 0 → x≥0, None → free


@dataclass
class ConstraintInfo:
    name: str
    expr: SymExpr                       # (LHS − RHS), parameterised by `forall`
    sense: str                          # '<=', '>=', '='
    forall: Tuple[Symbol, ...]
    forall_sets: Tuple[str, ...]
    multiplier_name: str
    multiplier_base: object             # IndexedBase or plain Symbol


# =============================================================================
# 3.  The DualDeriver
# =============================================================================


class DualDeriver:
    """
    Build an LP symbolically, construct its Lagrangian, derive the dual.

    Sign convention (primal min):
        constraint g(x) ≤ 0  →  +λ g(x)   (λ ≥ 0)
        constraint g(x) ≥ 0  →  −λ g(x)   (λ ≥ 0)
        constraint h(x) = 0  →  +ν h(x)   (ν free)
    Signs flip for a primal max.

    Dual-derivation logic (the only non-trivial part):
        • ∂L/∂v becomes a *dual constraint*, with sense determined by v's bound:
                v ≥ 0   ⇒  ∂L/∂v ≥ 0  (min primal)        / ≤ 0  (max)
                v free  ⇒  ∂L/∂v = 0
          (so that the infimum/supremum of L over v is finite)
        • The dual *objective* is the part of L containing no primal variable,
          because every primal-dependent term gets driven to zero at the
          dual-feasible optimum.
    """

    def __init__(self, sense: str = "min"):
        assert sense in ("min", "max")
        self.sense = sense
        self.variables: Dict[str, VarInfo] = {}
        self.constraints: List[ConstraintInfo] = []
        self.objective: Optional[SymExpr] = None

    # ---- declaration -------------------------------------------------------
    def declare_var(self, name: str,
                    index_sets: List[str],
                    free_symbols: Optional[List[Symbol]] = None,
                    lower: Optional[float] = 0) -> IndexedBase:
        base = IndexedBase(name)
        if free_symbols is None:
            free_symbols = [Symbol(s.split('_')[0].lower()) for s in index_sets]
        self.variables[name] = VarInfo(base, tuple(index_sets),
                                       tuple(free_symbols), lower)
        return base

    def set_objective(self, expr: SymExpr) -> None:
        self.objective = expr

    def add_constraint(self,
                       lhs: SymExpr, sense: str, rhs: SymExpr,
                       forall: List[Symbol], forall_sets: List[str],
                       multiplier_name: str,
                       name: Optional[str] = None) -> object:
        assert sense in ("<=", ">=", "=")
        forall = tuple(forall); forall_sets = tuple(forall_sets)
        mbase = IndexedBase(multiplier_name) if forall else Symbol(multiplier_name)
        self.constraints.append(ConstraintInfo(
            name or f"c{len(self.constraints)+1}",
            lhs - rhs, sense, forall, forall_sets, multiplier_name, mbase,
        ))
        return mbase

    # ---- Lagrangian --------------------------------------------------------
    def build_lagrangian(self) -> SymExpr:
        assert self.objective is not None, "Objective not set."
        L = SymExpr(self.objective.terms)
        for c in self.constraints:
            L = L + self._penalty(c)
        return L

    def _penalty(self, c: ConstraintInfo) -> SymExpr:
        if c.forall:
            mult = (c.multiplier_base[c.forall]
                    if len(c.forall) > 1 else c.multiplier_base[c.forall[0]])
        else:
            mult = c.multiplier_base

        # Sign that turns (LHS − RHS) sense 0  into  ± λ (LHS − RHS) in L
        if self.sense == "min":
            sign = +1 if c.sense == "<=" else (-1 if c.sense == ">=" else +1)
        else:
            sign = -1 if c.sense == "<=" else (+1 if c.sense == ">=" else +1)

        pen = c.expr.multiplied_by(mult)
        if sign < 0:
            pen = -pen
        for idx, s in zip(reversed(c.forall), reversed(c.forall_sets)):
            pen = pen.wrap_outer_sum(idx, s)
        return pen

    # ---- Dual --------------------------------------------------------------
    def derive_dual(self) -> Dict[str, object]:
        L = self.build_lagrangian()
        primal_bases = {v.base for v in self.variables.values()}
        mult_bases   = {c.multiplier_base for c in self.constraints if c.forall}

        # Dual objective: terms of L with no primal variable.
        dual_obj = L.parts_without(primal_bases)

        # Dual constraints from ∂L/∂v.
        dual_constraints = []
        for vname, vinfo in self.variables.items():
            deriv = L.diff_indexed(vinfo.base, vinfo.free_symbols)
            if vinfo.lower == 0:
                raw_sense = ">=" if self.sense == "min" else "<="
            elif vinfo.lower is None:
                raw_sense = "="
            else:
                raw_sense = ">="
            lhs, sense, rhs = self._rearrange(deriv, mult_bases, raw_sense)
            dual_constraints.append({
                "var":          vname,
                "raw_deriv":    deriv,
                "raw_sense":    raw_sense,
                "lhs":          lhs,
                "sense":        sense,
                "rhs":          rhs,
                "free_symbols": vinfo.free_symbols,
                "forall_sets":  vinfo.index_sets,
            })

        # Multiplier non-negativity (only for inequality primal constraints).
        mult_constraints = []
        for c in self.constraints:
            if c.sense in ("<=", ">="):
                if c.forall:
                    me = (c.multiplier_base[c.forall]
                          if len(c.forall) > 1 else c.multiplier_base[c.forall[0]])
                else:
                    me = c.multiplier_base
                mult_constraints.append({
                    "expr": me, "sense": ">=",
                    "forall": c.forall, "forall_sets": c.forall_sets,
                })

        return {
            "lagrangian":            L,
            "dual_sense":            "max" if self.sense == "min" else "min",
            "dual_objective_expr":   dual_obj,
            "dual_constraints":      dual_constraints,
            "multiplier_constraints": mult_constraints,
        }

    @staticmethod
    def _rearrange(expr: SymExpr, mult_bases: Set, sense: str):
        """
        Cosmetic rewrite of  ∂L/∂v sense 0  into  (mult side) sense (obj coeff).
        Splits ``expr`` into multiplier-containing and parameter-only SumTerms,
        moves the parameter-only ones to the RHS, then flips signs so the
        multiplier side has a majority of positive coefficients.
        """
        mult_terms, param_terms = [], []
        for t in expr.terms:
            bases = {a.base for a in t.body.atoms(Indexed)}
            (mult_terms if bases & mult_bases else param_terms).append(t)

        lhs = SymExpr(mult_terms)
        rhs = -SymExpr(param_terms)
        new_sense = sense

        pos = sum(1 for t in lhs.terms if t.sign > 0)
        neg = sum(1 for t in lhs.terms if t.sign < 0)
        if neg > pos:
            lhs = -lhs
            rhs = -rhs
            new_sense = {">=": "<=", "<=": ">=", "=": "="}[sense]
        return lhs, new_sense, rhs


# =============================================================================
# 4.  Pretty-printer for the derived dual
# =============================================================================


_SENSE_LATEX = {"<=": r"\leq", ">=": r"\geq", "=": "="}


def _forall_clause(syms, sets) -> str:
    if not syms:
        return ""
    parts = [rf"\forall\, {latex(d)} \in {latex(Symbol(s))}"
             for d, s in zip(syms, sets)]
    return r",\ \ " + r",\ ".join(parts)


def render_dual(result: Dict[str, object]) -> str:
    sense_word = "Maximize" if result["dual_sense"] == "max" else "Minimize"
    lines = [
        rf"\textbf{{{sense_word}}}\quad "
        + result["dual_objective_expr"].to_latex(),
        r"\text{subject to:}",
    ]
    for dc in result["dual_constraints"]:
        lhs = dc["lhs"].to_latex(); rhs = dc["rhs"].to_latex()
        op  = _SENSE_LATEX[dc["sense"]]
        lines.append("    " + lhs + " " + op + " " + rhs
                     + _forall_clause(dc["free_symbols"], dc["forall_sets"]))
    for mc in result["multiplier_constraints"]:
        lines.append("    " + latex(mc["expr"]) + r" \geq 0"
                     + _forall_clause(mc["forall"], mc["forall_sets"]))
    return "\n".join(lines)


# =============================================================================
# 5.  VALIDATION TEST (Ground-Truth Verification)
# =============================================================================
#
# Primal (minimise w.r.t. y_{jk} and varphi_i)
# ---------------------------------------------
#   min  Σ_{j∈J} Σ_{k∈K_j} c_{jk} y_{jk}  +  Σ_{i∈I} e_i varphi_i
#   s.t. Σ_{j∈J} Σ_{k∈K_j} a_{ijk} y_{jk}  +  varphi_i  ≥  d_i           ∀ i ∈ I
#        y_{jk}                              ≤  U_{jk} x_{jk}             ∀ j,k
#        y_{jk}                              ≥  L_{jk} x_{jk}             ∀ j,k
#        y_{jk}, varphi_i ≥ 0
#
# Expected dual (ground truth)
# ----------------------------
#   max  Σ_i λ_i d_i  −  Σ_{jk} μ_{jk} U_{jk} x_{jk}  +  Σ_{jk} ν_{jk} L_{jk} x_{jk}
#   s.t. λ_i                                            ≤  e_{i}          ∀ i ∈ I
#        Σ_i a_{ijk} λ_i  −  μ_{jk}  +  ν_{jk}          ≤  c_{jk}         ∀ j,k
#        λ, μ, ν ≥ 0
# =============================================================================


def _format(sx: SymExpr) -> str:
    """Render a SymExpr.canonical() map as a human-readable comparison list."""
    return "{\n  " + ",\n  ".join(
        f"({latex(b)} | dummies={dums} | sets={sets}) : {sign:+d}"
        for (b, dums, sets), sign in sx.canonical().items()
    ) + "\n}"


def _run_validation():
    deriver = DualDeriver(sense="min")

    # ---- Indices ---------------------------------------------------------
    i, j, k = sp.symbols("i j k")

    # ---- Parameters (constants in the dual derivation) -------------------
    c, a, U, L_, x, d, e = (IndexedBase(n) for n in "c a U L x d e".split())

    # ---- Decision variables ---------------------------------------------
    y   = deriver.declare_var("y",   ["J", "K_j"], free_symbols=[j, k], lower=0)
    phi = deriver.declare_var("varphi", ["I"],     free_symbols=[i],    lower=0)

    # ---- Objective -------------------------------------------------------
    deriver.set_objective(
          SUM(c[j, k] * y[j, k], (j, "J"), (k, "K_j"))
        + SUM(e[i]    * phi[i],  (i, "I"))
    )

    # ---- Constraints -----------------------------------------------------
    # (1) Σ_{j,k} a_{ijk} y_{jk}  +  φ_i  ≥  d_i,  ∀ i
    deriver.add_constraint(
        lhs   = SUM(a[i, j, k] * y[j, k], (j, "J"), (k, "K_j")) + TERM(phi[i]),
        sense = ">=",
        rhs   = TERM(d[i]),
        forall = [i], forall_sets = ["I"],
        multiplier_name = "lambda",
        name = "demand_cover",
    )
    # (2) y_{jk} ≤ U_{jk} x_{jk}, ∀ j,k
    deriver.add_constraint(
        lhs = TERM(y[j, k]), sense = "<=", rhs = TERM(U[j, k] * x[j, k]),
        forall = [j, k], forall_sets = ["J", "K_j"],
        multiplier_name = "mu", name = "upper_bound",
    )
    # (3) y_{jk} ≥ L_{jk} x_{jk}, ∀ j,k
    deriver.add_constraint(
        lhs = TERM(y[j, k]), sense = ">=", rhs = TERM(L_[j, k] * x[j, k]),
        forall = [j, k], forall_sets = ["J", "K_j"],
        multiplier_name = "nu", name = "lower_bound",
    )

    # ---- Derive ----------------------------------------------------------
    result = deriver.derive_dual()

    line = "=" * 80
    print(line)
    print("LAGRANGIAN  L(y, varphi, lambda, mu, nu)")
    print(line)
    print(result["lagrangian"].to_latex())

    print("\n" + line)
    print("DERIVED DUAL  (LaTeX)")
    print(line)
    print(render_dual(result))

    # ---- Ground-truth check ---------------------------------------------
    print("\n" + line)
    print("GROUND-TRUTH VERIFICATION")
    print(line)

    lam = IndexedBase("lambda")
    mu  = IndexedBase("mu")
    nu  = IndexedBase("nu")

    # (A) Dual objective
    expected_obj = SymExpr([
        SumTerm(lam[i]      * d[i],                    (i,),    ("I",),         +1),
        SumTerm(mu[j, k]    * U[j, k] * x[j, k],       (j, k),  ("J", "K_j"),   -1),
        SumTerm(nu[j, k]    * L_[j, k] * x[j, k],      (j, k),  ("J", "K_j"),   +1),
    ])
    derived_obj = result["dual_objective_expr"]
    assert derived_obj.canonical() == expected_obj.canonical(), (
        "Dual objective mismatch:\n"
        f"  derived:  {_format(derived_obj)}\n"
        f"  expected: {_format(expected_obj)}"
    )
    print("  [OK] Dual objective matches  Σ λ_i d_i  −  Σ μ_jk U_jk x_jk  +  Σ ν_jk L_jk x_jk")

    # (B) Dual constraint on φ:   λ_i ≤ e_i
    phi_dc = next(dc for dc in result["dual_constraints"] if dc["var"] == "varphi")
    assert phi_dc["sense"] == "<="
    assert phi_dc["lhs"].canonical() == SymExpr([SumTerm(lam[i])]).canonical()
    assert phi_dc["rhs"].canonical() == SymExpr([SumTerm(e[i])]).canonical()
    print("  [OK] ∂L/∂φ_i   gives   λ_i ≤ e_i")

    # (C) Dual constraint on y:   Σ_i a_ijk λ_i − μ_jk + ν_jk ≤ c_jk
    y_dc = next(dc for dc in result["dual_constraints"] if dc["var"] == "y")
    assert y_dc["sense"] == "<="
    expected_y_lhs = SymExpr([
        SumTerm(a[i, j, k] * lam[i], (i,), ("I",), +1),
        SumTerm(mu[j, k],            (),   (),     -1),
        SumTerm(nu[j, k],            (),   (),     +1),
    ])
    expected_y_rhs = SymExpr([SumTerm(c[j, k])])
    assert y_dc["lhs"].canonical() == expected_y_lhs.canonical(), (
        f"LHS mismatch:\n derived:  {_format(y_dc['lhs'])}\n"
        f" expected: {_format(expected_y_lhs)}"
    )
    assert y_dc["rhs"].canonical() == expected_y_rhs.canonical()
    print("  [OK] ∂L/∂y_jk  gives   Σ_i a_ijk λ_i − μ_jk + ν_jk ≤ c_jk")

    # (D) Multiplier non-negativity (3 inequality constraints in primal)
    sign_constraints = result["multiplier_constraints"]
    names = {str(mc["expr"].base) if isinstance(mc["expr"], Indexed)
             else str(mc["expr"]) for mc in sign_constraints}
    assert names == {"lambda", "mu", "nu"}
    print("  [OK] All three inequality multipliers λ, μ, ν are constrained ≥ 0")

    print("\n  All structural and algebraic checks PASSED.")
    print(line)
    return result


if __name__ == "__main__":
    _run_validation()
