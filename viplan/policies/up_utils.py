"""Module for working with the Unified Planning (UP) framework and PDDL.

This module provides utilities for handling PDDL (Planning Domain Definition Language)
problems, including state management, object handling, and data collection.
"""

import os
from itertools import product
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Callable

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import (
    Problem,
    UPState,
    FNode,
    Not,
    Action,
    InstantaneousAction,
    Fluent,
    Compiler,
    CompilationKind
)
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.plans.plan import ActionInstance


def has_quantifiers(problem: Problem) -> bool:
    if 'forall' in str(problem).lower() or 'exists' in str(problem).lower():
        return True
    return False


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
) -> List[FNode]:
    """Generate all possible grounded predicates for the given objects.
    
    Args:
        up_problem: The Unified Planning problem instance.
        objects: Optional dictionary mapping types to object names.
        
    Returns:
        List of all possible grounded predicate strings.
    """
    grounded_predicates = []
    for p in up_problem.fluents:
        varlists = []
        for variable in p.signature:
            varlists.append(list(up_problem.objects(variable.type)))
        for assignment in product(*varlists):
            grounded_predicates.append(p(*assignment))

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


def find_goal_relevant_fluents(problem: Problem) -> Set[FNode]:
    """
    Compute a set of (ground) fluent applications (atoms) that are backward-relevant
    to achieving the goals of the given unified_planning Problem.

    Result elements are FNodes of the form fluent(obj1, obj2, ...),
    e.g., at(r1, l2).

    Algorithm (backward-ish relevance):
    1. Start from atoms that appear in the goals.
    2. Repeatedly:
       - Mark an action as relevant if **any** fluent symbol occurring in:
         * its preconditions,
         * its (durative) conditions,
         * its effect targets,
         * its effect conditions,
         * its effect values
         is already relevant.
       - For each such relevant action, add all atoms that appear in all of
         those places (preconditions, conditions, effect targets, effect
         conditions, effect values).
    3. Stop when a fixpoint is reached (no new atoms added).

    This ensures that *preconditions* and *effect conditions* can make an
    action relevant, not only its effect targets.
    """
    # start by grounding the problem so that we only get grounded atoms
    with Compiler(problem_kind=problem.kind,
            compilation_kind=CompilationKind.QUANTIFIERS_REMOVING) as compiler:
        problem = compiler.compile(problem,
                                CompilationKind.QUANTIFIERS_REMOVING).problem
    # ground using fd
    with Compiler(name="fast-downward-grounder") as compiler:
        problem = compiler.compile(
            problem,
        ).problem

    env = problem.environment
    fve = env.free_vars_extractor  # extracts fluent applications from expressions

    # --- helpers -------------------------------------------------------------

    def add_atoms_from_expressions(target_atoms: Set[FNode],
                                   target_symbols: Set[Fluent],
                                   *expressions):
        """Collect all fluent applications (atoms) from the given expressions."""
        for e in expressions:
            if e is None:
                continue
            for atom in fve.get(e):
                if atom not in target_atoms:
                    target_atoms.add(atom)
                    target_symbols.add(atom.fluent())

    def action_touches_relevant(a, relevant_symbols: Set[Fluent]) -> bool:
        """
        Check whether this action mentions any currently-relevant fluent symbol
        in any of its effects. We only care about effects because we are checking
        if this action can produce relevant fluents (to add more relevant fluents).
        """
        for effect in a.effects:
            for atom in fve.get(effect.fluent):
                if atom.fluent() in relevant_symbols:
                    return True
        return False

    def add_all_atoms_from_action(a,
                                  target_atoms: Set[FNode],
                                  target_symbols: Set[Fluent]):
        """Once an action is relevant, add ALL atoms it mentions anywhere."""
        if isinstance(a, InstantaneousAction):
            # preconditions
            add_atoms_from_expressions(target_atoms, target_symbols, *a.preconditions)
            # effects: targets, conditions, values
            for eff in a.effects:
                add_atoms_from_expressions(
                    target_atoms,
                    target_symbols,
                    eff.condition,
                )

        # Extend similarly for Events / Processes if you use them.

    # --- initialization ------------------------------------------------------

    relevant_atoms: Set[FNode] = set()
    relevant_symbols: Set[Fluent] = set()

    # Seed relevance with atoms from the goals
    add_atoms_from_expressions(relevant_atoms, relevant_symbols, *problem.goals)

    # (Optional) include trajectory constraints / timed goals:
    # add_atoms_from_expressions(relevant_atoms, relevant_symbols,
    #                            *problem.trajectory_constraints)
    # for gl in problem.timed_goals.values():
    #     add_atoms_from_expressions(relevant_atoms, relevant_symbols, *gl)

    # --- fixpoint loop -------------------------------------------------------

    changed = True
    while changed:
        changed = False

        for a in problem.actions:
            # If this action touches any currently-relevant fluent in ANY role,
            # it becomes relevant and we add all atoms from it.
            if action_touches_relevant(a, relevant_symbols):
                before = len(relevant_atoms)
                add_all_atoms_from_action(a, relevant_atoms, relevant_symbols)
                if len(relevant_atoms) > before:
                    changed = True

    return relevant_atoms


def get_fd_constraints(domain_file: str, problem_file: str, unique_id: str) -> set:
    try:
        import up_fast_downward
    except ImportError:
        raise ImportError("up_fast_downward is required to compute reachable atoms. Please install it via 'pip install up-fast-downward'.")
    
    fd_path = os.path.join(os.path.dirname(up_fast_downward.__file__), 'downward', 'fast-downward.py')
    sas_file_path = f"output_{unique_id}.sas"
    cmd = [
        sys.executable,
        fd_path,
        '--translate',
        '--sas-file',
        sas_file_path,
        domain_file,
        problem_file,
    ]

    subprocess.run(cmd, check=True)

    exactly_one_groups = []
    at_most_one_groups = []
    all_reachable_atoms = set()
    mutex_groups = []

    # var_values[var_idx][val_idx] = atom_str or None
    var_values = []

    with open(sas_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # ------------------------------------------------------------------
    # First pass: parse variables, build your groups and var_values map
    # ------------------------------------------------------------------
    i = 0
    while i < len(lines):
        line = lines[i]

        if line == 'begin_variable':
            # Layout:
            # i     : begin_variable
            # i+1   : var_name
            # i+2   : axiom_layer
            # i+3   : range (num_values)
            # i+4.. : values (num_values lines)
            # last  : end_variable
            num_values = int(lines[i + 3])

            current_group = set()
            has_none_of_those = False
            value_list = []  # for var_values[var_idx]

            for k in range(num_values):
                val_line = lines[i + 4 + k]

                atom = None
                if val_line.startswith("Atom ") or val_line.startswith("NegatedAtom "):
                    # Strip "Atom " or "NegatedAtom " to get the predicate
                    atom = val_line.split(" ", 1)[1].strip()
                    current_group.add(atom)
                    all_reachable_atoms.add(atom)
                elif "<none of those>" in val_line:
                    has_none_of_those = True

                # Map this value index to its atom (or None)
                value_list.append(atom)

            # Categorize the group using your original logic
            if has_none_of_those:
                at_most_one_groups.append(current_group)
            else:
                exactly_one_groups.append(current_group)

            # Store mapping for later mutex parsing
            var_values.append(value_list)

            # Skip to the line after end_variable
            i += 4 + num_values + 1

        else:
            i += 1

    # ------------------------------------------------------------------
    # Second pass: parse explicit mutex groups (begin_mutex_group)
    # ------------------------------------------------------------------
    i = 0
    while i < len(lines):
        line = lines[i]

        if line == "begin_mutex_group":
            # Layout:
            # i     : begin_mutex_group
            # i+1   : N (number of entries)
            # i+2.. : "var_idx value_idx" N lines
            # last  : end_mutex_group
            n = int(lines[i + 1])
            group = set()

            for k in range(n):
                entry_line = lines[i + 2 + k]
                parts = entry_line.split()
                if len(parts) != 2:
                    continue

                v_idx = int(parts[0])
                val_idx = int(parts[1])

                # Defensive checks in case of weird indices
                if 0 <= v_idx < len(var_values) and 0 <= val_idx < len(var_values[v_idx]):
                    atom = var_values[v_idx][val_idx]
                    if atom is not None:
                        group.add(atom)

            mutex_groups.append(group)

            # Skip past this mutex group
            i += 2 + n + 1

        else:
            i += 1

    return {
        "exactly_one": exactly_one_groups,
        "at_most_one": at_most_one_groups,
        "reachable_atoms": all_reachable_atoms,
        "mutex_groups": mutex_groups,
    }

    

