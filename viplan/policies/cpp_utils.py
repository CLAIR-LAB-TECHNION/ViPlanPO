import heapq
from typing import List, Dict, Generator, Tuple, Iterable

from unified_planning.shortcuts import *
from unified_planning.model import ContingentProblem

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


def to_contingent_problem(problem: Problem) -> ContingentProblem:
    cp = ContingentProblem(f"Contingent_{problem.name}")

    problem = _do_required_compilations(problem)

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
        possible_init_states: Iterable[Dict[FNode, bool]]
) -> None:
    # clear existing initial state
    problem._initial_value.clear()
    problem._or_initial_constraints.clear()
    problem._oneof_initial_constraints.clear()

    # set all known fluents
    unknown_fluents = set()
    for f, v in possible_init_states[0].items():
        if all(v == s.get(f, None) for s in possible_init_states):
            problem.set_initial_value(f, v)
        else:
            unknown_fluents.add(f)

    # Encode as a disjunction of full-state conjunctions
    # only include fluents with unknown values
    formulas = []
    for s in possible_init_states:
        lits = [(f if s[f] else Not(f)) for f in unknown_fluents]
        if lits:
            formulas.append(And(*lits))
    if formulas:
        problem.add_oneof_initial_constraint(formulas)


def extract_conformant_plan(p_planNode):
    out = []
    while p_planNode is not None:
        out.append(p_planNode.action_instance)
        if len(p_planNode.children) > 0:  # there is only one child in conformant plans
            p_planNode = p_planNode.children[0][1]
        else:
            p_planNode = None
    return out


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