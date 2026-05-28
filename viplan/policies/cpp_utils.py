import heapq
from itertools import product as iproduct
from typing import List, Dict, Generator, Tuple, Iterable

from unified_planning.shortcuts import *
from unified_planning.model.contingent import ContingentProblem
from unified_planning.engines.results import PlanGenerationResultStatus

from .up_utils import has_quantifiers





def expand_quantifiers(problem: Problem, expr: FNode) -> FNode:
    # Very simplified: only handles Forall over a single variable v of some type T
    # and assumes the body has no *nested* quantifiers.
    env = problem.environment
    em = env.expression_manager

    if expr.is_forall():
        vars = expr.variables()           # <-- NOTE: variables() is a *method*
        assert len(vars) == 1, "helper currently only supports one quantified var"
        v = vars[0]
        body = expr.arg(0)

        objs = list(problem.objects(v.type))
        if not objs:
            # Forall over empty domain is True
            return em.TRUE()

        grounded_bodies = []
        for o in objs:
            # substitute v with object o; both are valid 'Expression's
            grounded = body.substitute({v: o})
            grounded_bodies.append(expand_quantifiers(problem, grounded))

        return em.And(grounded_bodies)

    # ---------- EXISTS ----------
    if expr.is_exists():
        vars = expr.variables()
        assert len(vars) == 1, "helper currently only supports one quantified var"
        v = vars[0]
        body = expr.arg(0)

        objs = list(problem.objects(v.type))
        if not objs:
            # Exists over empty domain is False
            return em.FALSE()

        grounded_bodies = []
        for o in objs:
            grounded = body.substitute({v: o})
            grounded_bodies.append(expand_quantifiers(grounded))

        return em.Or(grounded_bodies)

    # ---------- IMPLIES -> OR/NOT (to keep preconditions simple) ----------
    if expr.is_implies():
        left = expand_quantifiers(expr.arg(0))
        right = expand_quantifiers(expr.arg(1))
        return em.Or(em.Not(left), right)

    # ---------- BOOLEAN CONNECTIVES ----------
    if expr.is_and():
        return em.And([expand_quantifiers(problem, c) for c in expr.args])

    if expr.is_or():
        return em.Or([expand_quantifiers(problem, c) for c in expr.args])

    if expr.is_not():
        return em.Not(expand_quantifiers(problem, expr.arg(0)))

    # ---------- LEAF / OTHER NODES ----------
    return expr

def compile_precondition_quatifiers(problem: Problem) -> Problem:
    problem = problem.clone()
    for a in problem.actions:
        new_pres = [expand_quantifiers(problem, p) for p in a.preconditions]
        a.clear_preconditions()
        for p in new_pres:
            a.add_precondition(p)
    
    return problem


def _do_required_compilations(problem: Problem) -> Problem:
    if problem.kind.has_universal_conditions():
        print('removing quantifiers in preconditions...')
        problem = compile_precondition_quatifiers(problem)
    if has_quantifiers(problem):
        print('removing quantifiers...')
        with Compiler(problem_kind=problem.kind,
                    compilation_kind=CompilationKind.QUANTIFIERS_REMOVING) as compiler:
            problem = compiler.compile(problem,
                                    CompilationKind.QUANTIFIERS_REMOVING).problem
    if problem.kind.has_disjunctive_conditions():
        print('removing disjunctive conditions...')
        with Compiler(problem_kind=problem.kind,
                    compilation_kind=CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING) as compiler:
            problem = compiler.compile(problem,
                                    CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING).problem
    return problem


def _rewrite_negative_goals(problem: Problem) -> Problem:
    """Replace negative goal literals with positive auxiliary fluents.

    CPORLib's GetCNFClauses throws NotImplementedException on Not-formulas.
    For every Not(f(args)) in a goal, this introduces a fluent neg_f(args)
    whose value is always the negation of f(args), then rewrites the goal to
    use neg_f directly.
    """
    em = problem.environment.expression_manager
    problem = problem.clone()

    goals = list(problem.goals)
    problem.clear_goals()

    # Collect every fluent that appears negated in a goal
    neg_fluent_map: Dict = {}  # Fluent -> auxiliary Fluent

    def _scan(formula):
        if formula.is_not() and formula.arg(0).is_fluent_exp():
            fl = formula.arg(0).fluent()
            if fl not in neg_fluent_map:
                sig_kwargs = {p.name: p.type for p in fl.signature}
                neg_fluent_map[fl] = Fluent(f"neg_{fl.name}", fl.type, **sig_kwargs)
        for arg in formula.args:
            _scan(arg)

    for g in goals:
        _scan(g)

    if not neg_fluent_map:
        for g in goals:
            problem.add_goal(g)
        problem._neg_fluent_map = {}
        return problem

    # Register new fluents and set their initial values
    for fl, neg_fl in neg_fluent_map.items():
        problem.add_fluent(neg_fl, default_initial_value=False)
        obj_lists = [list(problem.objects(p.type)) for p in fl.signature]
        for combo in iproduct(*obj_lists):
            fa = fl(*combo)
            iv = problem.initial_value(fa)
            iv_bool = iv.bool_constant_value() if (iv is not None and iv.is_bool_constant()) else False
            problem.set_initial_value(neg_fl(*combo), not iv_bool)

    # Mirror every action effect that touches a rewritten fluent
    for action in problem.actions:
        new_effects = []
        for effect in action.effects:
            eff_fl = effect.fluent.fluent()
            if eff_fl in neg_fluent_map and effect.value.is_bool_constant():
                neg_fa = neg_fluent_map[eff_fl](*effect.fluent.args)
                new_effects.append((neg_fa, not effect.value.bool_constant_value(), effect.condition))
        for neg_fa, neg_val, cond in new_effects:
            action.add_effect(neg_fa, neg_val, cond)

    # Rewrite goal formulas
    def _rewrite(formula):
        if formula.is_not() and formula.arg(0).is_fluent_exp():
            fa = formula.arg(0)
            fl = fa.fluent()
            if fl in neg_fluent_map:
                return neg_fluent_map[fl](*fa.args)
        if formula.is_and():
            return em.And([_rewrite(a) for a in formula.args])
        if formula.is_or():
            return em.Or([_rewrite(a) for a in formula.args])
        if formula.is_not():
            return em.Not(_rewrite(formula.arg(0)))
        return formula

    for g in goals:
        problem.add_goal(_rewrite(g))

    problem._neg_fluent_map = neg_fluent_map  # consumed by set_cp_initial_state_constraints_from_belief
    return problem


def to_contingent_problem(problem: Problem) -> ContingentProblem:
    cp = ContingentProblem(f"Contingent_{problem.name}")

    problem = _do_required_compilations(problem)
    problem = _rewrite_negative_goals(problem)
    cp._neg_fluent_map = problem._neg_fluent_map  # propagate so set_cp_initial_state_constraints_from_belief can sync derived fluents

    # Objects
    cp.add_objects(problem.all_objects)

    # Fluents (preserve default values)
    for fl in problem.fluents:
        default = problem.fluents_defaults.get(fl, None)
        if default is None:
            cp.add_fluent(fl)
        else:
            cp.add_fluent(fl, default_initial_value=default)

    # Actions
    cp.add_actions(problem.actions)

    # Goals and constraints
    if len(problem.goals) == 1 and problem.goals[0].is_and():
        for g in problem.goals[0].args:
            cp.add_goal(g)
    else:
        for g in problem.goals:
            cp.add_goal(g)
    
    return cp


def set_cp_initial_state_constraints_from_belief(
        problem: ContingentProblem,
        possible_init_states: Iterable[Dict[FNode, bool]],
        version: int = 0,
) -> None:
    # clear existing initial state
    problem._initial_value.clear()
    problem._or_initial_constraints.clear()
    problem._oneof_initial_constraints.clear()

    # Fluent -> auxiliary neg_Fluent mapping produced by _rewrite_negative_goals
    neg_fluent_map = getattr(problem, '_neg_fluent_map', {})

    if version == 0:
        # set all known fluents
        unknown_fluents = set()
        for f, v in possible_init_states[0].items():
            if all(v == s.get(f, None) for s in possible_init_states):
                problem.set_initial_value(f, v)
                # keep the derived neg_fluent in sync
                fl = f.fluent()
                if fl in neg_fluent_map:
                    problem.set_initial_value(neg_fluent_map[fl](*f.args), not v)
            else:
                unknown_fluents.add(f)
    elif version == 1:
        unknown_fluents = list(possible_init_states[0].keys())
    else:
        raise ValueError(f"Unknown version {version} for setting initial state constraints")

    # Encode as a disjunction of full-state conjunctions.
    # Include derived neg_fluents in each conjunction so the planner
    # sees consistent values for auxiliary fluents across all belief states.
    formulas = []
    for s in possible_init_states:
        lits = []
        for f in unknown_fluents:
            lits.append(f if s[f] else Not(f))
            fl = f.fluent()
            if fl in neg_fluent_map:
                neg_f = neg_fluent_map[fl](*f.args)
                lits.append(neg_f if not s[f] else Not(neg_f))
        if lits:
            formulas.append(And(*lits))
    if formulas:
        problem.add_oneof_initial_constraint(formulas)

def set_cp_initial_state_without_constraints_from_belief(
        problem: ContingentProblem,
        possible_init_states: Iterable[Dict[FNode, bool]],
        version: int = 0,
) -> None:
    # clear existing initial state
    problem._initial_value.clear()

    for f, v in possible_init_states[0].items():
        problem.set_initial_value(f, v)


def extract_conformant_plan(p_planNode):
    out = []
    while p_planNode is not None:
        out.append(p_planNode.action_instance)
        if len(p_planNode.children) > 0:  # there is only one child in conformant plans
            p_planNode = p_planNode.children[0][1]
        else:
            p_planNode = None
    return out


def cpor_solve(problem: ContingentProblem,
               possible_init_states: Iterable[Dict[FNode, bool]],
               timeout: float,
               task_logger = None,
               log_plan_extra: Dict[str, str] = None):
    
    for i in range(2):
        # set initial state constraints in the contingent problem
        # based on all states selected so far.
        set_cp_initial_state_constraints_from_belief(
            problem,
            possible_init_states,
            version=i
        )

        # try to find a conformant plan for the largest belief set
        plan_res = None
        try:
            with OneshotPlanner(name="MetaCPORPlanning[fast-downward]") as planner:
                plan_res = planner.solve(
                    problem,
                    timeout=timeout
                )
        except Exception as e:
            if task_logger is not None:
                task_logger.error(
                    "Error during planning",
                    extra=log_plan_extra | {"exception": str(e)} if log_plan_extra else {"exception": str(e)}
                )
            else:
                print(f"Error during planning: {e}")

        if plan_res is not None and plan_res.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            return plan_res
    
    return None


def enumerate_states_by_probability(
        belief_state: Dict[FNode, float]
    ) -> Generator[Tuple[str, float], None, None]:
    """
    Enumerate all possible states based on the belief state, sorted by their probability.

    Args:
        belief_state: The current belief state of the system

    Yields:
        Strings representing boolean assignments
    """
    probs = list(belief_state.values())
    num_fluents = len(probs)
    
    def calculate_prob(assignment: Tuple[bool, ...]) -> float:
        """Calculate probability of an assignment."""
        prob = 1.0
        for i, value in enumerate(assignment):
            prob *= probs[i] if value else (1 - probs[i])
        return prob
    
    # Start with the most probable state
    initial_state = tuple(p > 0.5 for p in probs)
    initial_prob = calculate_prob(initial_state)
    
    # Priority queue: (-prob, assignment) - negative for max-heap
    pq = [(-initial_prob, initial_state)]
    visited = {initial_state}
    count = 0
    
    while pq:
        neg_prob, current = heapq.heappop(pq)
        
        # Yield this state
        yield ''.join('1' if val else '0' for val in current), -neg_prob
        count += 1
        
        # Generate neighbors by flipping each bit
        for i in range(num_fluents):
            # Create neighbor by flipping bit i
            neighbor = list(current)
            neighbor[i] = not neighbor[i]
            neighbor = tuple(neighbor)
            
            if neighbor not in visited:
                visited.add(neighbor)
                #TODO update probability using FD constraints
                neighbor_prob = calculate_prob(neighbor)
                heapq.heappush(pq, (-neighbor_prob, neighbor))
