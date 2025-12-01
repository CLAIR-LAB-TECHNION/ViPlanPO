"""Module for working with the Unified Planning (UP) framework and PDDL.

This module provides utilities for handling PDDL (Planning Domain Definition Language)
problems, including state management, object handling, and data collection.
"""

import re
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Callable

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import Problem, UPState, FNode, Not, Action
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.plans.plan import ActionInstance


def create_up_problem(domain: str, problem: str) -> Problem:
    """Create a Unified Planning problem from PDDL files or strings.
    
    Args:
        domain: PDDL domain file path or string.
        problem: PDDL problem file path or string.
        
    Returns:
        A Unified Planning problem instance.
        
    Raises:
        AssertionError: If domain is a file but problem is not.
    """
    reader = PDDLReader()
    if domain.lower().endswith(".pddl"):
        assert problem.lower().endswith(
            ".pddl"
        ), "if domain is a file, problem must also be a file"
        up_problem = reader.parse_problem(domain, problem)
    else:
        up_problem = reader.parse_problem_string(domain, problem)

    return up_problem


def get_object_names_dict(up_problem: Problem) -> Dict[str, List[str]]:
    """Get a dictionary mapping object types to lists of object names.
    
    Args:
        up_problem: The Unified Planning problem instance.
        
    Returns:
        Dictionary mapping type names to lists of object names.
    """
    objects = {}
    for t in up_problem.user_types:
        objects[t.name] = list(map(str, up_problem.objects(t)))

    return objects


def get_all_grounded_predicates_for_objects(
    up_problem: Problem, objects: Optional[Dict[str, List[str]]] = None
) -> List[str]:
    """Generate all possible grounded predicates for the given objects.
    
    Args:
        up_problem: The Unified Planning problem instance.
        objects: Optional dictionary mapping types to object names.
        
    Returns:
        List of all possible grounded predicate strings.
    """
    predicates = up_problem.fluents
    if objects is None:
        objects = get_object_names_dict(up_problem)

    grounded_predicates = []
    for p in predicates:
        varlists = []
        for variable in p.signature:
            varlists.append(objects[variable.type.name])
        for assignment in product(*varlists):
            grounded_predicates.append(f'{p.name}({",".join(assignment)})')

    return grounded_predicates


def get_pddl_files_str(up_problem: Problem) -> Tuple[str, str]:
    """Convert a Unified Planning problem to PDDL strings.
    
    Args:
        up_problem: The Unified Planning problem instance.
        
    Returns:
        Tuple of (domain string, problem string).
    """
    writer = PDDLWriter(up_problem)
    return writer.get_domain(), writer.get_problem()


def ground_predicate_str_to_fnode(up_problem: Problem, predicate_str: str) -> FNode:
    """Convert a grounded predicate string to a Unified Planning FNode.
    
    Args:
        up_problem: The Unified Planning problem instance.
        predicate_str: The grounded predicate string.
        
    Returns:
        A Unified Planning FNode representing the predicate.
    """
    fluent_name, args = predicate_str.split("(")
    args = args.rstrip(")").split(",")
    args = [arg.strip() for arg in args if arg]
    pred_obj = up_problem.fluent(fluent_name)
    arg_obj = [up_problem.object(a) for a in args]
    if arg_obj:
        return pred_obj(*arg_obj)
    else:
        return pred_obj()


def bool_constant_to_fnode(up_problem: Problem, constant: bool) -> FNode:
    """Convert a boolean constant to a Unified Planning FNode.
    
    Args:
        up_problem: The Unified Planning problem instance.
        constant: The boolean value to convert.
        
    Returns:
        A Unified Planning FNode representing the boolean constant.
    """
    exp_mgr = up_problem.environment.expression_manager
    if constant is True:
        return exp_mgr.true_expression
    else:
        return exp_mgr.false_expression


def convert_state_dict_to_up_compatible(
    up_problem, state_dict: Dict[str, bool]
) -> Dict[FNode, FNode]:
    """Convert a state dictionary to Unified Planning compatible format.
    
    Args:
        up_problem: The Unified Planning problem instance.
        state_dict: Dictionary mapping predicate strings to boolean values.
        
    Returns:
        Dictionary mapping FNodes to boolean FNodes.
    """
    return {
        ground_predicate_str_to_fnode(up_problem, k): bool_constant_to_fnode(
            up_problem, v
        )
        for k, v in state_dict.items()
    }


def state_dict_to_up_state(up_problem: Problem, state_dict: Dict[str, bool]) -> UPState:
    """Convert a state dictionary to a Unified Planning state.
    
    Args:
        up_problem: The Unified Planning problem instance.
        state_dict: Dictionary mapping predicate strings to boolean values.
        
    Returns:
        A Unified Planning state.
    """
    return UPState(convert_state_dict_to_up_compatible(up_problem, state_dict))


def up_state_to_state_dict(up_state: UPState) -> Dict[str, bool]:
    """Convert a Unified Planning state to a state dictionary.
    
    Args:
        up_state: The Unified Planning state to convert.
        
    Returns:
        Dictionary mapping predicate strings to boolean values.
    """
    current_instance = up_state
    out = {}
    while current_instance is not None:
        for k, v in current_instance._values.items():
            out.setdefault(
                f'{k.fluent().name}({",".join(map(str, k.args))})', v.constant_value()
            )
        current_instance = current_instance._father

    return out


def set_problem_init_state(up_problem: Problem, init_state_dict: Dict[str, bool]):
    """Set the initial state of a Unified Planning problem.
    
    Args:
        up_problem: The Unified Planning problem instance.
        init_state_dict: Dictionary mapping predicate strings to boolean values.
    """
    # clear existing fluents
    up_problem.explicit_initial_values.clear()

    # set desired fluents
    for k, v in convert_state_dict_to_up_compatible(
        up_problem, init_state_dict
    ).items():
        up_problem.set_initial_value(k, v)


def set_problem_goal_state(
    up_problem: Problem, goal_state_dict: Dict[str, bool], include_negatives=False
):
    """Set the goal state of a Unified Planning problem.
    
    Args:
        up_problem: The Unified Planning problem instance.
        goal_state_dict: Dictionary mapping predicate strings to boolean values.
        include_negatives: Whether to include negative goals.
    """
    # clear existing goals
    up_problem.clear_goals()

    # set desired goals
    for k, v in goal_state_dict.items():
        if v is True:
            up_problem.add_goal(
                ground_predicate_str_to_fnode(up_problem, k),
            )
        elif include_negatives:
            up_problem.add_goal(
                Not(ground_predicate_str_to_fnode(up_problem, k)),
            )

def equality_to_predicate(domain_path: str, problem_path: str, output_path: str):
    with open(domain_path, 'r') as f:
        domain_pddl = f.read()
    with open(problem_path, 'r') as f:
        problem_pddl = f.read()
    
    # if ":equality" in domain_pddl:
    #   add (equal ?x ?y) predicate to the domain if not already present
    #   remove all occurrences of (= ?x ?y) in the domain PDDL
    #   add (equal x x) in the problem initial state for objects x
    if ":equality" in domain_pddl:
        if "(equal" not in domain_pddl:
            # add (equal ?x ?y) predicate to the domain
            pred_str = "(equal ?x - object ?y - object)"
            pred_index = domain_pddl.find("(:predicates")
            if pred_index != -1:
                pred_end = domain_pddl.find("s", pred_index) + 1
                domain_pddl = domain_pddl[:pred_end] + "\n    " + pred_str + domain_pddl[pred_end:]
        
        # remove all occurrences of (= ?x ?y) in the domain PDDL
        domain_pddl = domain_pddl.replace("(= ", "(equal ")
        
        # add (equal x x) in the problem initial state for objects x
        init_index = problem_pddl.find("(:init")
        if init_index != -1:
            init_end = problem_pddl.find("t", init_index) + 1
            objects_start = problem_pddl.find("(:objects")
            objects_end = problem_pddl.find(")", objects_start) + 1
            objects_str = problem_pddl[objects_start:objects_end]
            object_lines = objects_str.splitlines()[1:-1]
            object_names = []

            for line in object_lines:
                parts = line.split()
                object_names.extend(parts[:-2])  # exclude type
            equalities = "\n        ".join([f"(equal {obj} {obj})" for obj in object_names])
            new_init = "\n    " + equalities 
            problem_pddl = problem_pddl[:init_end] + new_init + problem_pddl[init_end:]

    # write the modified domain and problem to output_path
    with open('domain_' + output_path, 'w') as f:
        f.write(domain_pddl)
    with open('problem_' + output_path, 'w') as f:
        f.write(problem_pddl)


def get_mapping_from_compiled_actions_to_original_actions(
    compiled_problem: Problem, original_problem: Problem
) -> Callable[[ActionInstance], ActionInstance]:
    # create a dictionary mapping compiled actions to original actions
    action_mapping = {}
    original_action_names = [action.name for action in original_problem.actions]
    for action in compiled_problem.actions:
        action_name = action.name
        for original_action_name in original_action_names:
            if action_name.startswith(original_action_name):
                action_mapping[action] = original_problem.action(original_action_name)
                break
    
    def map_compiled_action_to_original_action(
        compiled_action_instance: ActionInstance
    ) -> ActionInstance:
        original_action = action_mapping[compiled_action_instance.action]
        params_original_problem = tuple()
        for param in compiled_action_instance.actual_parameters:
            obj = original_problem.object(param.object().name)
            params_original_problem += (obj,)
        return ActionInstance(original_action, params_original_problem)
    
    return map_compiled_action_to_original_action
