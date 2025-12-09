import heapq
from typing import List, Dict, Generator, Tuple, Iterable

from unified_planning.shortcuts import *
from unified_planning.model import ContingentProblem


def _do_required_compilations(problem: Problem) -> Problem:
    if problem.kind.has_universal_conditions():
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