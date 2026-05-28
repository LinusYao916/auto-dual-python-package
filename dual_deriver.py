"""
dual_deriver.py
================

Automated Lagrangian / Dual derivation tool built on SymPy.

Handles two classes of primal:

    (a) Linear programs                       — derivative ∂L/∂v is constant
        in v, so each primal variable yields a *dual constraint*.
    (b) Strictly convex quadratic programs    — derivative ∂L/∂v still
        contains v, so we **solve** ∂L/∂v = 0 for v* and **substitute** v*
        back into L to get a (possibly quadratic) *dual objective*.

DESIGN: I/O FORMAT
------------------
A primal model is built declaratively against a ``DualDeriver`` instance.

    deriver = DualDeriver(sense='min')

    i, j, k = sp.symbols('i j k')
    c, a, U, L, x, d, e_hat, beta = (sp.IndexedBase(s) for s in
        'c a U L x d e_hat beta'.split())

    y      = deriver.declare_var('y',     ['J','K'],  free_symbols=[j,k], lower=0)
    varphi = deriver.declare_var('varphi',['I'],      free_symbols=[i],   lower=None)

    deriver.set_objective(
        SUM(c[j,k]*y[j,k],         (j,'J'),(k,'K'))
      + SUM(e_hat[i]*varphi[i],    (i,'I'))
      + SUM(beta[i]*varphi[i]**2,  (i,'I'))          #  ←  quadratic term
    )
    deriver.add_constraint(
        lhs   = SUM(a[i,j,k]*y[j,k],(j,'J'),(k,'K')) + TERM(varphi[i]),
        sense = '>=',  rhs = TERM(d[i]),
        forall = [i], forall_sets = ['I'],
        multiplier_name = 'lambda',
    )
    ...
    result = deriver.derive_dual()
    print(render_dual(result))

Sugar:
  ``SUM(body, (dummy, set_name), ...)``  →  Σ_{dummy ∈ set_name} body
  ``TERM(body)``                         →  body  (no summation)

Both return a ``SymExpr`` (a list of ``SumTerm``s).  ``+``, ``-``, unary ``-``,
multiplication by an Indexed factor, and Python's ``**`` on individual bodies
are all supported through the underlying SymPy expressions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set

import sympy as sp
from sympy import IndexedBase, Indexed, Symbol, S, latex, Add, Mul


# =============================================================================
# 1.  Internal symbolic algebra:  SumTerm  and  SymExpr
# =============================================================================


@dataclass
class SumTerm:
    """
    sign · Σ_{d_1 ∈ S_1, ..., d_n ∈ S_n}  body

    `body` is an ordinary SymPy expression that may use both the dummies and
    any free outer index symbols.  No restriction to linear bodies — powers
    and arbitrary nonlinear sub-expressions are fine.
    """
    body: sp.Expr
    dummies: Tuple[Symbol, ...] = ()
    sets:    Tuple[str, ...]    = ()
    sign:    int                = 1

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
        """Expand and split so each SumTerm holds one monomial-ish body."""
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
        body_latex = latex(self.body)
        # CRITICAL: wrap multi-addend bodies in parens whenever there is a
        # non-trivial prefix on the LEFT (a Σ or a leading "−"), so the
        # prefix binds to the WHOLE body and not just its first addend.
        # Without this, a SumTerm with sign=-1 and body=(a+b+c) would render
        # as "- a + b + c"  ==  −a + b + c  (WRONG: the minus only eats the
        # first addend).  Must render as "- (a + b + c)".
        if isinstance(self.body, Add) and (self.dummies or self.sign < 0):
            body_latex = r"\left(" + body_latex + r"\right)"
        sums = "".join(rf"\sum_{{{latex(d)} \in {latex(Symbol(s))}}} "
                       for d, s in zip(self.dummies, self.sets))
        return f"{sigil} {sums}{body_latex}"


class SymExpr:
    """An additive combination of SumTerms."""

    def __init__(self, terms: Optional[Iterable[SumTerm]] = None):
        self.terms: List[SumTerm] = []
        for t in (terms or []):
            self.terms.extend(t.split_addends())

    @classmethod
    def from_raw(cls, terms: Iterable[SumTerm]) -> "SymExpr":
        """Build a SymExpr WITHOUT re-running split_addends (preserves complex
        bodies such as `(λ-e)^2 / (4β)` produced by consolidation)."""
        obj = cls.__new__(cls)
        obj.terms = list(terms)
        return obj

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
    def diff_indexed(self, base,
                     free: Tuple[Symbol, ...]) -> "SymExpr":
        """
        Symbolic partial derivative wrt the primal variable ``base`` (which
        may be either an ``IndexedBase`` -- in which case ``free`` lists the
        index symbols at which we differentiate -- or a plain ``Symbol`` for
        a scalar variable, in which case ``free`` is ignored).

        Works for *arbitrary* nonlinear bodies via ``sp.diff``.  For Indexed
        primals, each Indexed occurrence of ``base`` in a SumTerm body is
        α-renamed (dummy ↦ free) before differentiating, and the unconsumed
        dummies stay as summation indices in the result.
        """
        out: List[SumTerm] = []
        target = _target_for(base, free)

        # ---- scalar (plain Symbol) primal ---------------------------------
        if not isinstance(base, IndexedBase):
            for term in self.terms:
                if not term.body.has(target):
                    continue
                deriv_body = _safe_diff(term.body, target)
                if deriv_body == 0:
                    continue
                out.append(SumTerm(deriv_body, term.dummies, term.sets, term.sign))
            return SymExpr(out)

        # ---- Indexed primal -----------------------------------------------
        for term in self.terms:
            for occ in term.body.atoms(Indexed):
                if occ.base != base:
                    continue
                occ_idx = tuple(occ.indices)
                if len(occ_idx) != len(free):
                    continue

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
                        ok = False; break
                if not ok:
                    continue

                renamed_body = term.body.subs(subs_map)
                deriv_body   = _safe_diff(renamed_body, target)

                rem_dummies = tuple(d for d in term.dummies
                                    if d not in subs_map)
                rem_sets    = tuple(s for d, s in zip(term.dummies, term.sets)
                                    if d not in subs_map)

                out.append(SumTerm(deriv_body, rem_dummies, rem_sets, term.sign))
        return SymExpr(out)

    def substitute_indexed(self, base,
                           free: Tuple[Symbol, ...],
                           solution: sp.Expr) -> "SymExpr":
        """
        Replace every occurrence of the primal variable ``base`` with the
        supplied ``solution``.

        For an Indexed ``base``, every ``base[any_indices]`` is replaced and
        ``free`` is aligned to the actual indices each occurrence carries.
        For a plain Symbol ``base`` (scalar variable), ``free`` is ignored
        and the Symbol is substituted directly in every body.
        """
        new_terms: List[SumTerm] = []
        if not isinstance(base, IndexedBase):
            for t in self.terms:
                body = t.body.subs(base, solution)
                new_terms.append(SumTerm(body, t.dummies, t.sets, t.sign))
            return SymExpr(new_terms)

        for t in self.terms:
            body = t.body
            for occ in list(body.atoms(Indexed)):
                if occ.base != base:
                    continue
                occ_idx = tuple(occ.indices)
                if len(occ_idx) != len(free):
                    continue
                subs_map = dict(zip(free, occ_idx))
                sol_inst = solution.subs(subs_map, simultaneous=True)
                body = body.subs(occ, sol_inst)
            new_terms.append(SumTerm(body, t.dummies, t.sets, t.sign))
        return SymExpr(new_terms)

    def parts_without(self, var_bases: Set) -> "SymExpr":
        """Keep only the SumTerms whose body uses none of ``var_bases`` (each
        element of which may be an ``IndexedBase`` or a plain ``Symbol``)."""
        keep = []
        for t in self.terms:
            bases = _bases_in_body(t.body)
            if bases.isdisjoint(var_bases):
                keep.append(t)
        return SymExpr(keep)

    def consolidate(self) -> "SymExpr":
        """
        Merge SumTerms that share the same (dummies, sets) by summing their
        bodies (with signs) and ``sp.simplify``-ing the result -- producing
        one compact SumTerm per Σ-shape.  Groups containing a single SumTerm
        are passed through untouched so well-typed LP terms aren't disturbed.
        """
        groups: Dict[Tuple, List[SumTerm]] = {}
        for t in self.terms:
            groups.setdefault((t.dummies, t.sets), []).append(t)

        out: List[SumTerm] = []
        for (dummies, sets), terms in groups.items():
            if len(terms) == 1:
                out.append(terms[0]); continue
            body_sum = S.Zero
            for t in terms:
                body_sum = body_sum + t.sign * t.body
            body_s = sp.simplify(sp.together(body_sum))
            if body_s == 0:
                continue
            if body_s.could_extract_minus_sign():
                out.append(SumTerm(-body_s, dummies, sets, -1))
            else:
                out.append(SumTerm(body_s,  dummies, sets, +1))
        return SymExpr.from_raw(out)

    # -- comparison / display ------------------------------------------------
    def canonical(self) -> Dict[Tuple, int]:
        out: Dict[Tuple, int] = {}
        for t in self.terms:
            key = (sp.expand(t.body), t.dummies, t.sets)
            out[key] = out.get(key, 0) + t.sign
        return {k: v for k, v in out.items() if v != 0}

    def has_base(self, base) -> bool:
        if isinstance(base, IndexedBase):
            return any(occ.base == base
                       for t in self.terms for occ in t.body.atoms(Indexed))
        # Scalar Symbol primal
        return any(base in t.body.atoms(Symbol) for t in self.terms)

    def to_latex(self) -> str:
        if not self.terms:
            return "0"
        s = " ".join(t.to_latex() for t in self.terms).strip()
        return s[1:].lstrip() if s.startswith("+") else s


def _target_for(base, free: Tuple[Symbol, ...]) -> sp.Expr:
    """Build the symbolic 'point' we differentiate / substitute at.

    Indexed primal   →  base[free]   (an ``Indexed`` expression)
    Scalar Symbol    →  base         (the symbol itself; ``free`` ignored)
    """
    if not isinstance(base, IndexedBase):
        return base
    return base[free] if len(free) > 1 else base[free[0]]


def _bases_in_body(body: sp.Expr) -> Set:
    """All variable/multiplier bases used in an expression body -- whether
    they appear as ``IndexedBase`` (via Indexed atoms) or as plain Symbols.
    Returns the union, which is safe to intersect against either kind."""
    return {a.base for a in body.atoms(Indexed)} | body.atoms(Symbol)


def _safe_diff(expr: sp.Expr, target: sp.Expr) -> sp.Expr:
    """``sp.diff`` works with Indexed targets in recent SymPy, but route
    through a Dummy substitution so we don't depend on that quirk."""
    if not expr.has(target):
        # Linear/constant w.r.t. target  →  derivative is the linear coefficient.
        return expr.coeff(target) if hasattr(expr, "coeff") else S.Zero
    tmp = sp.Dummy("_d")
    return sp.diff(expr.subs(target, tmp), tmp).subs(tmp, target)


def _safe_solve(eq: sp.Expr, target: sp.Expr) -> Optional[sp.Expr]:
    """Solve ``eq = 0`` for ``target`` (which may be an Indexed)."""
    if eq == 0 or not eq.has(target):
        return None
    tmp = sp.Dummy("_q")
    eq_t = eq.subs(target, tmp)
    sols = sp.solve(eq_t, tmp)
    if not sols:
        return None
    sol = sols[0] if isinstance(sols, list) else (
        next(iter(sols.values())) if isinstance(sols, dict) else sols
    )
    return sol.subs(tmp, target)


def SUM(body: sp.Expr, *index_set_pairs) -> SymExpr:
    """SUM(c[j,k]*y[j,k], (j,'J'), (k,'K_j'))  →  Σ_{j∈J} Σ_{k∈K_j} c·y"""
    dummies = tuple(p[0] for p in index_set_pairs)
    sets    = tuple(p[1] for p in index_set_pairs)
    return SymExpr([SumTerm(body, dummies, sets)])


def TERM(body: sp.Expr) -> SymExpr:
    """Non-summed term -- parameterised only by surrounding free indices."""
    return SymExpr([SumTerm(body)])


# =============================================================================
# 2.  Model declaration containers
# =============================================================================


@dataclass
class VarInfo:
    base: object                        # IndexedBase  OR  plain Symbol (scalar)
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
    Build a primal model symbolically, build its Lagrangian, derive the dual.

    Sign convention (primal min):
        constraint g(x) ≤ 0   →  +λ g(x)   (λ ≥ 0)
        constraint g(x) ≥ 0   →  −λ g(x)   (λ ≥ 0)
        constraint h(x) = 0   →  +ν h(x)   (ν free)
    Signs flip for a primal max.

    Dual-derivation pipeline:
        1.  For every primal variable v, compute ∂L/∂v.
        2.  If ∂L/∂v still depends on v (the QP/strictly-convex case), solve
            ∂L/∂v = 0 analytically for v* and substitute back into L.  No
            dual constraint is produced for such a variable; its quadratic
            contribution is folded into the dual objective.
        3.  For variables that survive (∂L/∂v doesn't depend on v), generate
            a *dual constraint*:
                  v ≥ 0   (min primal)  →  ∂L/∂v ≥ 0
                  v free                →  ∂L/∂v = 0
        4.  The dual objective is the part of (post-substitution) L that
            contains none of the surviving primal variables, consolidated by
            Σ-shape so the quadratic compactly reads, e.g., -(λ-e)^2/(4β).
    """

    def __init__(self, sense: str = "min"):
        assert sense in ("min", "max")
        self.sense = sense
        self.variables:   Dict[str, VarInfo]        = {}
        self.constraints: List[ConstraintInfo]      = []
        self.objective:   Optional[SymExpr]         = None

    # ---- declaration -------------------------------------------------------
    def declare_var(self, name: str,
                    index_sets: Optional[List[str]] = None,
                    free_symbols: Optional[List[Symbol]] = None,
                    lower: Optional[float] = 0):
        """Declare a primal variable.

        * ``index_sets=None`` or ``[]``  →  scalar variable
          (returned as a plain ``sympy.Symbol``; ``free_symbols`` ignored).
        * Otherwise                       →  indexed variable
          (returned as a ``sympy.IndexedBase``; ``free_symbols`` defaults to
          lowercased first chars of the set names).
        """
        if not index_sets:
            base = Symbol(name)
            self.variables[name] = VarInfo(base, (), (), lower)
            return base

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
                       forall: Optional[List[Symbol]] = None,
                       forall_sets: Optional[List[str]] = None,
                       multiplier_name: str = "lambda",
                       name: Optional[str] = None) -> object:
        assert sense in ("<=", ">=", "=")
        forall      = tuple(forall or ())
        forall_sets = tuple(forall_sets or ())
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

        # ---- Pass 1: detect strictly-convex primals and eliminate them ----
        eliminated: Dict[str, Dict[str, object]] = {}
        for vname, vinfo in self.variables.items():
            deriv = L.diff_indexed(vinfo.base, vinfo.free_symbols)
            if not deriv.has_base(vinfo.base):
                continue                              # linear → handled later
            target = _target_for(vinfo.base, vinfo.free_symbols)
            # Only the *local* (non-summed) part of ∂L/∂v can contain v at
            # the free index; that's what we need for ∂L/∂v(free) = 0.
            local_expr = S.Zero
            for t in deriv.terms:
                if not t.dummies:
                    local_expr = local_expr + t.sign * t.body
            sol = _safe_solve(local_expr, target)
            if sol is None:
                continue
            eliminated[vname] = {"vinfo": vinfo, "target": target,
                                 "solution": sol, "deriv": deriv}

        # ---- Pass 2: substitute eliminated v* back into L ------------------
        L_sub = L
        for vname, rec in eliminated.items():
            L_sub = L_sub.substitute_indexed(
                rec["vinfo"].base, rec["vinfo"].free_symbols, rec["solution"],
            )

        # ---- Pass 3: dual constraints from remaining (linear) primals -----
        remaining_bases = {v.base for vn, v in self.variables.items()
                           if vn not in eliminated}
        # Include BOTH indexed and scalar multipliers so _rearrange / dual-obj
        # filtering correctly classify scalar-multiplier terms.
        mult_bases = {c.multiplier_base for c in self.constraints}

        dual_constraints = []
        for vname, vinfo in self.variables.items():
            if vname in eliminated:
                continue
            deriv = L_sub.diff_indexed(vinfo.base, vinfo.free_symbols)
            if vinfo.lower == 0:
                raw_sense = ">=" if self.sense == "min" else "<="
            elif vinfo.lower is None:
                raw_sense = "="
            else:
                raw_sense = ">="
            lhs, sense, rhs = self._rearrange(deriv, mult_bases, raw_sense)
            dual_constraints.append({
                "var": vname,
                "raw_deriv":    deriv,
                "raw_sense":    raw_sense,
                "lhs": lhs, "sense": sense, "rhs": rhs,
                "free_symbols": vinfo.free_symbols,
                "forall_sets":  vinfo.index_sets,
            })

        # ---- Pass 4: multiplier non-negativity (inequality constraints) ----
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

        # ---- Pass 5: dual objective ---------------------------------------
        # IMPORTANT: filter out surviving-primal terms first, then consolidate
        # by Σ-shape -- consolidating first would mix v-containing and
        # v-free SumTerms in the same group and would force us to drop the
        # whole group.
        dual_obj_raw = L_sub.parts_without(remaining_bases)
        dual_obj = dual_obj_raw.consolidate() if eliminated else dual_obj_raw

        return {
            "lagrangian":             L,
            "lagrangian_substituted": L_sub,
            "eliminated":             eliminated,
            "dual_sense":             "max" if self.sense == "min" else "min",
            "dual_objective_expr":    dual_obj,
            "dual_constraints":       dual_constraints,
            "multiplier_constraints": mult_constraints,
        }

    # Convenience: callable as method or module function -- the user's
    # script does `deriver.render_dual(result)`, this resolves it.
    def render_dual(self, result: Dict[str, object]) -> str:
        return render_dual(result)

    @staticmethod
    def _rearrange(expr: SymExpr, mult_bases: Set, sense: str):
        """Cosmetic rewrite of (∂L/∂v sense 0) → (mult side) sense (obj coeff)."""
        mult_terms, param_terms = [], []
        for t in expr.terms:
            bases = _bases_in_body(t.body)
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
    if result["eliminated"]:
        lines.append(r"\text{(eliminated by KKT stationarity:)}")
        for vname, rec in result["eliminated"].items():
            lines.append(f"    {latex(rec['target'])} = {latex(rec['solution'])}")
    return "\n".join(lines)


# =============================================================================
# 5.  VALIDATION TESTS
# =============================================================================
#
# Test 1 -- LP (the original ground truth, unchanged):
# ----------------------------------------------------
#   min  Σ_{j,k} c_{jk} y_{jk}  +  Σ_i e_i varphi_i
#   s.t. Σ_{j,k} a_{ijk} y_{jk} + varphi_i ≥ d_i, ∀ i
#        y_{jk} ≤ U_{jk} x_{jk}, ∀ j,k
#        y_{jk} ≥ L_{jk} x_{jk}, ∀ j,k
#        y, varphi ≥ 0
#   Expected dual: λ_i ≤ e_i,  Σ_i a_ijk λ_i − μ_jk + ν_jk ≤ c_jk, λ,μ,ν ≥ 0.
#
# Test 2 -- QP (the new strictly-convex case):
# --------------------------------------------
#   min  Σ_{j,k} c_{jk} y_{jk}  +  Σ_i ê_i varphi_i  +  Σ_i β_i varphi_i²
#   s.t. (same three constraints)
#        y_{jk} ≥ 0;  varphi_i FREE              ← so KKT stationarity gives a
#                                                   unique interior optimum.
#   Expected dual objective per i:  λ_i d_i − (λ_i − ê_i)² / (4 β_i)
#   Expected dual constraint:       Σ_i a_ijk λ_i − μ_jk + ν_jk ≤ c_jk
#                                   (no constraint on λ_i ≤ ê_i because varphi
#                                    is no longer linear → no LP-style bound)
# =============================================================================


def _run_lp_validation():
    print("=" * 80)
    print("TEST 1  --  LINEAR PROGRAM  (original ground truth)")
    print("=" * 80)

    deriver = DualDeriver(sense="min")
    i, j, k = sp.symbols("i j k")
    c, a, U, L_, x, d, e = (IndexedBase(n) for n in "c a U L x d e".split())

    y   = deriver.declare_var("y",      ["J", "K_j"], free_symbols=[j, k], lower=0)
    phi = deriver.declare_var("varphi", ["I"],        free_symbols=[i],    lower=0)

    deriver.set_objective(
          SUM(c[j, k] * y[j, k], (j, "J"), (k, "K_j"))
        + SUM(e[i]    * phi[i],  (i, "I"))
    )
    deriver.add_constraint(
        lhs   = SUM(a[i, j, k] * y[j, k], (j, "J"), (k, "K_j")) + TERM(phi[i]),
        sense = ">=", rhs = TERM(d[i]),
        forall = [i], forall_sets = ["I"],
        multiplier_name = "lambda",
    )
    deriver.add_constraint(
        lhs = TERM(y[j, k]), sense = "<=", rhs = TERM(U[j, k] * x[j, k]),
        forall = [j, k], forall_sets = ["J", "K_j"],
        multiplier_name = "mu",
    )
    deriver.add_constraint(
        lhs = TERM(y[j, k]), sense = ">=", rhs = TERM(L_[j, k] * x[j, k]),
        forall = [j, k], forall_sets = ["J", "K_j"],
        multiplier_name = "nu",
    )

    result = deriver.derive_dual()
    print("\nDerived dual (LaTeX):")
    print(render_dual(result))

    # ground-truth check
    lam = IndexedBase("lambda"); mu = IndexedBase("mu"); nu = IndexedBase("nu")
    expected_obj = SymExpr([
        SumTerm(lam[i]   * d[i],                       (i,),  ("I",),       +1),
        SumTerm(mu[j, k] * U[j, k] * x[j, k],          (j, k),("J", "K_j"), -1),
        SumTerm(nu[j, k] * L_[j, k] * x[j, k],         (j, k),("J", "K_j"), +1),
    ])
    assert result["dual_objective_expr"].canonical() == expected_obj.canonical()
    print("  [OK] LP dual objective matches.")

    phi_dc = next(dc for dc in result["dual_constraints"] if dc["var"] == "varphi")
    assert phi_dc["sense"] == "<="
    assert phi_dc["lhs"].canonical() == SymExpr([SumTerm(lam[i])]).canonical()
    assert phi_dc["rhs"].canonical() == SymExpr([SumTerm(e[i])]).canonical()
    print("  [OK] LP φ-constraint    matches  λ_i ≤ e_i.")

    y_dc = next(dc for dc in result["dual_constraints"] if dc["var"] == "y")
    assert y_dc["sense"] == "<="
    assert y_dc["lhs"].canonical() == SymExpr([
        SumTerm(a[i, j, k] * lam[i], (i,), ("I",), +1),
        SumTerm(mu[j, k],            (),   (),     -1),
        SumTerm(nu[j, k],            (),   (),     +1),
    ]).canonical()
    assert y_dc["rhs"].canonical() == SymExpr([SumTerm(c[j, k])]).canonical()
    print("  [OK] LP y-constraint    matches  Σ_i a_ijk λ_i − μ_jk + ν_jk ≤ c_jk.")
    print()


def _run_qp_validation():
    print("=" * 80)
    print("TEST 2  --  STRICTLY-CONVEX QUADRATIC PROGRAM  (new)")
    print("=" * 80)

    deriver = DualDeriver(sense="min")
    i, j, k = sp.symbols("i j k")
    c, a, U, L_, x, d                = (IndexedBase(n) for n in "c a U L x d".split())
    e_hat, beta                       = IndexedBase("e_hat"), IndexedBase("beta")

    # varphi is FREE — the quadratic guarantees a bounded interior optimum.
    y      = deriver.declare_var("y",      ["J", "K"], free_symbols=[j, k], lower=0)
    varphi = deriver.declare_var("varphi", ["I"],      free_symbols=[i],   lower=None)

    deriver.set_objective(
          SUM(c[j, k]    * y[j, k],         (j, "J"), (k, "K"))
        + SUM(varphi[i]  * e_hat[i],        (i, "I"))
        + SUM(beta[i]    * varphi[i] ** 2,  (i, "I"))         # ← quadratic
    )
    deriver.add_constraint(
        lhs   = SUM(a[i, j, k] * y[j, k], (j, "J"), (k, "K")) + TERM(varphi[i]),
        sense = ">=", rhs = TERM(d[i]),
        forall = [i], forall_sets = ["I"],
        multiplier_name = "lambda",
    )
    deriver.add_constraint(
        lhs = TERM(y[j, k]), sense = "<=", rhs = TERM(U[j, k] * x[j, k]),
        forall = [j, k], forall_sets = ["J", "K"],
        multiplier_name = "mu",
    )
    deriver.add_constraint(
        lhs = TERM(y[j, k]), sense = ">=", rhs = TERM(L_[j, k] * x[j, k]),
        forall = [j, k], forall_sets = ["J", "K"],
        multiplier_name = "nu",
    )

    result = deriver.derive_dual()
    print("\nDerived dual (LaTeX):")
    print(render_dual(result))

    lam = IndexedBase("lambda"); mu = IndexedBase("mu"); nu = IndexedBase("nu")

    # (A) Stationarity gave varphi_i = (λ_i − ê_i) / (2 β_i)
    assert "varphi" in result["eliminated"], "varphi should be eliminated"
    sol = result["eliminated"]["varphi"]["solution"]
    expected_sol = (lam[i] - e_hat[i]) / (2 * beta[i])
    assert sp.simplify(sol - expected_sol) == 0
    print("  [OK] Stationarity gave  varphi_i = (λ_i − ê_i) / (2 β_i).")

    # (B) Dual objective: i-group should equal λ_i d_i − (λ_i − ê_i)² / (4 β_i)
    dual_obj = result["dual_objective_expr"]
    i_terms = [t for t in dual_obj.terms
               if t.dummies == (i,) and t.sets == ("I",)]
    i_body = sum((t.sign * t.body for t in i_terms), S.Zero)
    expected_i = lam[i] * d[i] - (lam[i] - e_hat[i]) ** 2 / (4 * beta[i])
    assert sp.simplify(i_body - expected_i) == 0, (
        f"i-group does not match.\n  got      {sp.simplify(i_body)}\n"
        f"  expected {sp.simplify(expected_i)}"
    )
    print("  [OK] Quadratic dual objective i-group  =  λ_i d_i − (λ_i − ê_i)² / (4 β_i).")

    # (C) Dual objective: (j,k)-group should equal ν_jk L_jk x_jk − μ_jk U_jk x_jk
    jk_terms = [t for t in dual_obj.terms
                if t.dummies == (j, k) and t.sets == ("J", "K")]
    jk_body = sum((t.sign * t.body for t in jk_terms), S.Zero)
    expected_jk = nu[j, k] * L_[j, k] * x[j, k] - mu[j, k] * U[j, k] * x[j, k]
    assert sp.simplify(jk_body - expected_jk) == 0
    print("  [OK] Dual objective (j,k)-group        =  ν_jk L_jk x_jk − μ_jk U_jk x_jk.")

    # (D) Dual constraint for y is unchanged (still linear)
    y_dc = next(dc for dc in result["dual_constraints"] if dc["var"] == "y")
    assert y_dc["sense"] == "<="
    assert y_dc["lhs"].canonical() == SymExpr([
        SumTerm(a[i, j, k] * lam[i], (i,), ("I",), +1),
        SumTerm(mu[j, k],            (),   (),     -1),
        SumTerm(nu[j, k],            (),   (),     +1),
    ]).canonical()
    assert y_dc["rhs"].canonical() == SymExpr([SumTerm(c[j, k])]).canonical()
    print("  [OK] y-dual constraint  Σ_i a_ijk λ_i − μ_jk + ν_jk ≤ c_jk  unchanged.")

    # (E) No spurious LP-style φ-constraint
    assert not any(dc["var"] == "varphi" for dc in result["dual_constraints"]), \
        "varphi shouldn't generate a dual constraint -- it was eliminated."
    print("  [OK] No spurious λ_i ≤ ê_i constraint  (varphi correctly eliminated).")

    # (F) End-to-end algebraic sanity:
    #     plug the substituted L back at a numeric point and check it equals
    #     the dual objective evaluated at the same multipliers.
    numeric = {lam[i]: sp.Rational(3),
               e_hat[i]: sp.Rational(1),
               beta[i]: sp.Rational(2),
               d[i]: sp.Rational(5)}
    # value of i-group:
    i_val   = float(i_body.xreplace(numeric))
    # value computed directly from the closed form:
    closed  = 3*5 - (3 - 1)**2 / (4 * 2)
    assert abs(i_val - closed) < 1e-12
    print(f"  [OK] Numeric spot-check       (i-group at λ=3,ê=1,β=2,d=5)  =  {i_val} = {closed}.")
    print()


def _run_minimal_qp_validation():
    """The user's exact minimal QP -- single variable, single constraint.

        min  Σ_i β_i varphi_i²
        s.t. varphi_i ≥ d_i,  ∀ i        (multiplier λ_i ≥ 0)
             varphi_i FREE
        Expected dual:  max Σ_i [λ_i d_i  −  λ_i² / (4 β_i)],   λ_i ≥ 0
    """
    print("=" * 80)
    print("TEST 3  --  MINIMAL QP  (the user's exact reproducer)")
    print("=" * 80)

    deriver = DualDeriver(sense="min")
    i = sp.symbols("i")
    beta, d = IndexedBase("beta"), IndexedBase("d")
    varphi = deriver.declare_var(
        name="varphi", index_sets=["I"], free_symbols=[i], lower=None,
    )

    deriver.set_objective(SUM(beta[i] * (varphi[i] ** 2), (i, "I")))
    deriver.add_constraint(
        lhs   = TERM(varphi[i]), sense = ">=", rhs = TERM(d[i]),
        forall = [i], forall_sets = ["I"], multiplier_name = "lambda",
    )

    result = deriver.derive_dual()
    print("\nDerived dual (LaTeX):")
    print(deriver.render_dual(result))                  # method form, as in the bug report

    # (A) Stationarity gave   varphi_i* = λ_i / (2 β_i)
    lam = IndexedBase("lambda")
    sol = result["eliminated"]["varphi"]["solution"]
    assert sp.simplify(sol - lam[i] / (2 * beta[i])) == 0, \
        f"stationary solution wrong: got {sol}"
    print(f"\n  [OK] Stationarity: varphi_i*  =  {sp.latex(sol)}.")

    # (B) Dual objective body must contain the quadratic /(4 β_i) penalty
    body = sum((t.sign * t.body for t in result["dual_objective_expr"].terms),
               S.Zero)
    expected = lam[i] * d[i] - lam[i] ** 2 / (4 * beta[i])
    assert sp.simplify(body - expected) == 0, (
        f"Quadratic dropped!  got = {sp.simplify(body)}"
    )
    print(f"  [OK] Dual objective body         =  {sp.latex(sp.simplify(body))}.")
    print(f"       (matches the expected       =  {sp.latex(expected)})")

    # (C) lambda must be present quadratically in the dual objective, with a
    #     coefficient that contains 1/β -- the failure mode in the bug report.
    quad_coeff = sp.simplify(body.coeff(lam[i], 2))
    assert quad_coeff != 0, "λ² coefficient is zero -- quadratic WAS dropped!"
    assert (1 / beta[i] in sp.Mul.make_args(quad_coeff)
            or sp.together(quad_coeff).has(beta[i])), \
        f"λ² coefficient does not contain 1/β: {quad_coeff}"
    print(f"  [OK] Coefficient of λ_i²         =  {quad_coeff} "
          f"(contains 1/β_i, NOT zero).")
    print()


def _run_scalar_validation():
    """Pure scalar QP -- no indices anywhere.

        min  x²                       (scalar variable, FREE)
        s.t. x ≥ d                    (scalar constraint, mult λ ≥ 0)
        Expected dual:  max  λ d  −  λ² / 4   s.t. λ ≥ 0
    """
    print("=" * 80)
    print("TEST 4  --  SCALAR QP  (no indices, no Σ)")
    print("=" * 80)

    deriver = DualDeriver(sense="min")
    x = deriver.declare_var("x", lower=None)            # scalar  → Symbol
    assert isinstance(x, Symbol), "scalar declare_var must return a Symbol"

    d = sp.Symbol("d")
    deriver.set_objective(TERM(x ** 2))
    deriver.add_constraint(
        lhs=TERM(x), sense=">=", rhs=TERM(d),
        multiplier_name="lambda",                       # forall/forall_sets defaulted
    )

    result = deriver.derive_dual()
    print("\nDerived dual (LaTeX):")
    print(deriver.render_dual(result))

    lam = sp.Symbol("lambda")

    # (A) Stationarity gave  x* = λ/2
    assert "x" in result["eliminated"], "scalar x should be eliminated"
    sol = result["eliminated"]["x"]["solution"]
    assert sp.simplify(sol - lam / 2) == 0, \
        f"scalar stationary solution wrong: got {sol}"
    print(f"\n  [OK] Stationarity: x*  =  {sp.latex(sol)}.")

    # (B) Dual objective body must equal  λ d − λ² / 4
    body = sum((t.sign * t.body for t in result["dual_objective_expr"].terms),
               S.Zero)
    expected = lam * d - lam ** 2 / 4
    assert sp.simplify(body - expected) == 0, (
        f"scalar dual objective wrong:\n  got      = {sp.simplify(body)}\n"
        f"  expected = {sp.simplify(expected)}"
    )
    print(f"  [OK] Dual objective body  =  {sp.latex(sp.simplify(body))}.")

    # (C) λ ≥ 0 still listed as a multiplier constraint
    assert any(mc["expr"] == lam for mc in result["multiplier_constraints"]), \
        "scalar λ ≥ 0 constraint missing"
    print(f"  [OK] λ ≥ 0 retained as multiplier constraint.")
    print()


if __name__ == "__main__":
    _run_lp_validation()
    _run_qp_validation()
    _run_minimal_qp_validation()
    _run_scalar_validation()
    print("=" * 80)
    print("ALL TESTS PASSED.")
    print("=" * 80)
