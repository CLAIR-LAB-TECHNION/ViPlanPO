import os

# set PYTHONNET_RUNTIME=mono
os.environ["PYTHONNET_RUNTIME"] = "mono"

# set PYTHONNET_MONO_LIBMONO="$(brew --prefix mono)/lib/libmonosgen-2.0.dylib"
os.environ["PYTHONNET_MONO_LIBMONO"] = (
    "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from copy import copy
import glob
from itertools import product, chain, combinations
import pytest

import unified_planning.environment as environment
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines.sequential_simulator import UPSequentialSimulator

from viplan.policies.up_utils import (
    get_mapping_from_compiled_actions_to_original_actions,
    ground_predicate_str_to_fnode,
    create_up_problem,
    get_all_grounded_predicates_for_objects,
)
from viplan.policies.cpp_utils import (
    to_contingent_problem,
    set_cp_initial_state_constraints_from_belief,
    cpor_solve
)


env = environment.get_environment()
env.factory.add_meta_engine("MetaCPORPlanning", "up_cpor.engine", "CPORMetaEngineImpl")


IGIBSON_PDDL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/planning/igibson"
)
ALL_DOMAIN_FILES = glob.glob(os.path.join(IGIBSON_PDDL_DIR, "*.pddl"))
ALL_PROBLEM_FILES = glob.glob(os.path.join(IGIBSON_PDDL_DIR, "*", "*.pddl"))
DOMAIN_FILE_COND = "data/planning/igibson/domain-cond.pddl"
PROBLEM_FILE_DRAWERS_SIMPLE = 'data/planning/igibson/simple/cleaning_out_drawers_simple.pddl'


DRAWERS_POSSIBLE_STATES_STR = [
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    }, 
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": True,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, sink_1)": False,
    },
    {
        "inside(bowl_1, cabinet_1)": False,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": True,
        "ontop(bowl_1, cabinet_1)": True,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
    },
]

DRAWERS_TRUE_INIT_STATE_STR = {
        "inside(bowl_1, cabinet_1)": True,
        "open(cabinet_1)": False,
        "ontop(bowl_1, sink_1)": False,
        "reachable(bowl_1)": False,
        "reachable(cabinet_1)": False,
        "reachable(sink_1)": True,
        "holding(bowl_1)": False,
        "ontop(bowl_1, bowl_1)": False,
        "ontop(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, bowl_1)": True,
        "nextto(bowl_1, cabinet_1)": False,
        "nextto(bowl_1, sink_1)": False,
}


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def extract_conformant_plan(p_planNode):
    out = []
    while p_planNode is not None:
        out.append(p_planNode.action_instance)
        if len(p_planNode.children) > 0:  # there is only one child in conformant plans
            p_planNode = p_planNode.children[0][1]
        else:
            p_planNode = None
    return out


@pytest.mark.parametrize("domain_file,problem_file",
                         product(ALL_DOMAIN_FILES, ALL_PROBLEM_FILES))
def test_as_classical_and_small_sets(domain_file, problem_file):
    problem = create_up_problem(domain_file, problem_file)

    ground_predicates = get_all_grounded_predicates_for_objects(problem)

    s1 = {}
    for f, v in problem.initial_values.items():
        if f.is_fluent_exp():
            s1[f] = v.constant_value()
        else:
            s1[f.args[0]] = not v.constant_value()
    for f in ground_predicates:
        if f not in s1:
            # default to False if not specified.
            # this is important because otherwise the planner
            # views it as unknown and may not be able to plan.
            s1[f] = False  

    cp = to_contingent_problem(problem)
    
    # set_cp_initial_state_constraints_from_belief(cp, [s1])
    # with OneshotPlanner(name="MetaCPORPlanning[fast-downward]") as planner:
    #     result = planner.solve(cp)
    #     print(result)
    #     assert result.plan is not None, f"Planner returned no plan for problem:\n{cp}"
    #     conformant_plan = extract_conformant_plan(result.plan.root_node)
    result = cpor_solve(
        cp,
        [s1],
        timeout=10.0
    )
    print(result)
    assert result is not None and result.plan is not None, f"Planner returned no plan for problem:\n{cp}"
    conformant_plan = extract_conformant_plan(result.plan.root_node)

    orig_problem = create_up_problem(domain_file, problem_file)
    actions_mapping = get_mapping_from_compiled_actions_to_original_actions(
        cp, orig_problem
    )

    action_seq_orig_problem = [actions_mapping(a) for a in conformant_plan]

    sim = UPSequentialSimulator(orig_problem)
    cur_state = sim.get_initial_state()
    for a in action_seq_orig_problem:
        print("Applying action:", a)
        new_state = sim.apply(cur_state, a)
        assert new_state is not None, f"Action {a} is not applicable in state {cur_state}"
        cur_state = new_state

    assert sim.is_goal(cur_state), "didn't reach the goal!"

    # this is the domain file where we edited it to have conditional effects
    # instead of some of the more strict preconditions.
    # here we should be able to test more states
    if not "cond" in domain_file:
        return

    s2 = copy(s1)
    keys_list = list(s1.keys())
    s2[keys_list[-1]] = not s2[keys_list[-1]]  # flip one fluent to create a different state
    s2[keys_list[-2]] = not s2[keys_list[-2]]  # flip another fluent
    
    # set_cp_initial_state_constraints_from_belief(cp, [s1, s2])
    # with OneshotPlanner(name="MetaCPORPlanning[fast-downward]") as planner:
    #     result = planner.solve(cp)
    #     print(result)
    #     assert result.plan is not None, f"Planner returned no plan for problem:\n{cp}"
    #     conformant_plan = extract_conformant_plan(result.plan.root_node)
    result = cpor_solve(
        cp,
        [s1],
        timeout=10.0
    )
    print(result)
    assert result is not None and result.plan is not None, f"Planner returned no plan for problem:\n{cp}"
    conformant_plan = extract_conformant_plan(result.plan.root_node)

    sim = UPSequentialSimulator(orig_problem)
    cur_state = sim.get_initial_state()
    for a in action_seq_orig_problem:
        print("Applying action:", a)
        new_state = sim.apply(cur_state, a)
        assert new_state is not None, f"Action {a} is not applicable in state {cur_state}"
        cur_state = new_state

    assert sim.is_goal(cur_state), "didn't reach the goal!"


@pytest.mark.parametrize('subset', list(powerset(DRAWERS_POSSIBLE_STATES_STR))[:200])
def test_subsets_of_initial_states(subset):
    problem = create_up_problem(DOMAIN_FILE_COND, PROBLEM_FILE_DRAWERS_SIMPLE)

    states_str = list(subset) + [DRAWERS_TRUE_INIT_STATE_STR]
    
    cp = to_contingent_problem(problem)

    states = []
    for state_str_dict in states_str:
        state = {}
        for f_str, val in state_str_dict.items():
            fnode = ground_predicate_str_to_fnode(cp, f_str)
            state[fnode] = val
        states.append(state)

    # set_cp_initial_state_constraints_from_belief(cp, states)
    # with OneshotPlanner(name="MetaCPORPlanning[fast-downward]") as planner:
    #     result = planner.solve(cp)
    #     print(result)
    #     assert result.plan is not None, f"Planner returned no plan for subset:\n{subset}\n\nproblem definition:\n{cp}"
    #     conformant_plan = extract_conformant_plan(result.plan.root_node)
    result = cpor_solve(
        cp,
        states,
        timeout=10.0
    )
    print(result)
    assert result is not None and result.plan is not None, f"Planner returned no plan for problem:\n{cp}"
    conformant_plan = extract_conformant_plan(result.plan.root_node)


    
    orig_problem = create_up_problem(DOMAIN_FILE_COND, PROBLEM_FILE_DRAWERS_SIMPLE)
    actions_mapping = get_mapping_from_compiled_actions_to_original_actions(
        cp, orig_problem
    )

    action_seq_orig_problem = [actions_mapping(a) for a in conformant_plan]

    sim = UPSequentialSimulator(orig_problem)
    sim = UPSequentialSimulator(orig_problem)
    cur_state = sim.get_initial_state()
    for a in action_seq_orig_problem:
        print("Applying action:", a)
        new_state = sim.apply(cur_state, a)
        assert new_state is not None, f"Action {a} is not applicable in state {cur_state}"
        cur_state = new_state

    assert sim.is_goal(cur_state), f"didn't reach the goal!, failed subset:\n{subset}\n\nproblem definition:\n{cp}"


