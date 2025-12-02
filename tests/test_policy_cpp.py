import logging
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch


sys.path.append(str(Path(__file__).resolve().parents[1]))


def _register(module_name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(module_name, module)


# Minimal torch stub to bypass heavy dependency imports during testing.
fake_torch = types.ModuleType("torch")
fake_torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace())
fake_torch.Tensor = type("FakeTensor", (), {})
_register("torch", fake_torch)
_register("torch.nn", fake_torch.nn)
_register("torch.nn.functional", fake_torch.nn.functional)
fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *_, **__: None)
fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *_, **__: None)
fake_transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *_, **__: None)
fake_transformers.GenerationConfig = SimpleNamespace
fake_transformers.AutoModelForImageTextToText = SimpleNamespace(
    from_pretrained=lambda *_, **__: None
)
fake_transformers.Gemma3ForConditionalGeneration = SimpleNamespace(
    from_pretrained=lambda *_, **__: None
)
_register("transformers", fake_transformers)
fake_mloggers = types.ModuleType("mloggers")
class _FakeConsoleLogger:
    def __init__(self, *_, **__):
        pass


class _FakeLogLevel:
    INFO = "info"


fake_mloggers.ConsoleLogger = _FakeConsoleLogger
fake_mloggers.LogLevel = _FakeLogLevel
fake_mloggers.FileLogger = _FakeConsoleLogger
fake_mloggers.MultiLogger = _FakeConsoleLogger
_register("mloggers", fake_mloggers)
class _FakeChatCompletion:
    def __init__(self):
        self.choices = [
            SimpleNamespace(logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=[])]))
        ]


fake_openai = types.ModuleType("openai")
fake_openai_types = types.ModuleType("openai.types")
fake_openai_types_chat = types.ModuleType("openai.types.chat")
fake_openai_types_chat_completion = types.ModuleType("openai.types.chat.chat_completion")
fake_openai_types_chat_completion.ChatCompletion = _FakeChatCompletion
fake_openai.types = fake_openai_types
fake_openai.OpenAI = lambda *_, **__: SimpleNamespace(
    chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **__: _FakeChatCompletion()))
)
_register("openai", fake_openai)
_register("openai.types", fake_openai_types)
_register("openai.types.chat", fake_openai_types_chat)
_register("openai.types.chat.chat_completion", fake_openai_types_chat_completion)
_register("requests", types.ModuleType("requests"))
fake_hf_hub = types.ModuleType("huggingface_hub")
fake_hf_hub.hf_hub_download = lambda *_, **__: ""
_register("huggingface_hub", fake_hf_hub)
fake_vllm = types.ModuleType("vllm")
fake_vllm.LLM = type("FakeLLM", (), {})
_register("vllm", fake_vllm)
fake_vllm_sampling = types.ModuleType("vllm.sampling_params")
fake_vllm_sampling.SamplingParams = type("SamplingParams", (), {})
_register("vllm.sampling_params", fake_vllm_sampling)
_register("anthropic", types.ModuleType("anthropic"))
fake_google = types.ModuleType("google")
fake_google_genai = types.ModuleType("google.genai")
fake_google_genai.Client = type("GenAIClient", (), {})
fake_google_genai.types = types.SimpleNamespace()
_register("google", fake_google)
_register("google.genai", fake_google_genai)


# Stub PIL so policy_cpp imports succeed without installing pillow.
fake_pil_module = types.ModuleType("PIL")
fake_pil_image_module = types.ModuleType("PIL.Image")
fake_pil_image_ops = types.ModuleType("PIL.ImageOps")


class _FakeImage:
    """Placeholder to satisfy PolicyCPP imports without pillow."""


fake_pil_image_module.Image = _FakeImage
_register("PIL", fake_pil_module)
_register("PIL.Image", fake_pil_image_module)
_register("PIL.ImageOps", fake_pil_image_ops)

# Stub unified_planning to avoid heavyweight dependencies during import.
fake_up_module = types.ModuleType("unified_planning")
fake_up_shortcuts = types.ModuleType("unified_planning.shortcuts")
fake_up_environment = types.ModuleType("unified_planning.environment")
fake_up_engines = types.ModuleType("unified_planning.engines")
fake_up_results = types.ModuleType("unified_planning.engines.results")
fake_up_model = types.ModuleType("unified_planning.model")
fake_up_io = types.ModuleType("unified_planning.io")
fake_up_sequential_simulator = types.ModuleType(
    "unified_planning.engines.sequential_simulator"
)
fake_up_plans = types.ModuleType("unified_planning.plans")
fake_up_plans_plan = types.ModuleType("unified_planning.plans.plan")


class _FakeFNode:
    """Placeholder for Unified Planning nodes."""


class _FakeOneshotPlanner:
    def __init__(self, *_, **__):
        pass


class _FakeUPSequentialSimulator:
    def __init__(self, *_, **__):
        pass


class _FakeActionInstance:
    def __init__(self, action=None, params=None):
        self.action = action
        self.actual_parameters = params or ()


class _FakePlanGenerationStatus:
    SOLVED_SATISFICING = "SOLVED_SATISFICING"


class _FakePDDLReader:
    def parse_problem(self, *_):
        return _FakeProblem()


class _FakePDDLWriter:
    def __init__(self, *_):
        pass

    def get_domain(self):
        return "(define (domain fake))"

    def get_problem(self):
        return "(define (problem fake))"


class _FakeContingentProblem:
    def __init__(self, *_, **__):
        pass

    def add_objects(self, *_):
        pass

    def add_fluent(self, *_ , **__):
        pass

    def add_actions(self, *_):
        pass

    def add_goal(self, *_):
        pass

    def set_initial_value(self, *_):
        pass

    def add_oneof_initial_constraint(self, *_):
        pass


class _FakeProblem:
    def __init__(self, *_, **__):
        self.name = "fake"


class _FakeCompiler:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        return False

    def compile(self, problem, *_):
        return SimpleNamespace(problem=problem)


class _FakeCompilationKind:
    QUANTIFIERS_REMOVING = "quant"
    DISJUNCTIVE_CONDITIONS_REMOVING = "disj"


class _FakeUPState:
    def __init__(self, *_, **__):
        self._values = {}
        self._father = None


def _fake_not(value):
    return ("not", value)


def _fake_and(*values):
    return ("and", values)


class _FakeAction:
    def __init__(self, *_, **__):
        self.name = "action"


def _get_environment_stub():
    return MagicMock(factory=MagicMock(add_meta_engine=MagicMock()))


fake_up_shortcuts.FNode = _FakeFNode
fake_up_shortcuts.OneshotPlanner = _FakeOneshotPlanner
fake_up_shortcuts.Problem = _FakeProblem
fake_up_shortcuts.Compiler = _FakeCompiler
fake_up_shortcuts.CompilationKind = _FakeCompilationKind
fake_up_shortcuts.UPState = _FakeUPState
fake_up_shortcuts.Not = _fake_not
fake_up_shortcuts.And = _fake_and
fake_up_shortcuts.Action = _FakeAction
fake_up_io.PDDLReader = _FakePDDLReader
fake_up_io.PDDLWriter = _FakePDDLWriter
fake_up_environment.get_environment = _get_environment_stub
fake_up_results.PlanGenerationResultStatus = _FakePlanGenerationStatus
fake_up_model.ContingentProblem = _FakeContingentProblem
fake_up_model.Problem = _FakeProblem
fake_up_sequential_simulator.UPSequentialSimulator = _FakeUPSequentialSimulator
fake_up_plans_plan.ActionInstance = _FakeActionInstance

fake_up_engines.results = fake_up_results
fake_up_engines.sequential_simulator = fake_up_sequential_simulator
fake_up_module.model = fake_up_model
fake_up_plans.plan = fake_up_plans_plan

_register("unified_planning", fake_up_module)
_register("unified_planning.shortcuts", fake_up_shortcuts)
_register("unified_planning.environment", fake_up_environment)
_register("unified_planning.io", fake_up_io)
_register("unified_planning.model", fake_up_model)
_register("unified_planning.engines", fake_up_engines)
_register("unified_planning.engines.results", fake_up_results)
_register("unified_planning.engines.sequential_simulator", fake_up_sequential_simulator)
_register("unified_planning.plans", fake_up_plans)
_register("unified_planning.plans.plan", fake_up_plans_plan)

# Stub numpy and scipy.special for lightweight math functionality.


class _FakeNumpy(types.ModuleType):
    def clip(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def prod(self, values):
        result = 1
        for val in values:
            result *= val
        return result


def _fake_logit(probability):
    return math.log(probability / (1 - probability))


def _fake_expit(value):
    return 1 / (1 + math.exp(-value))


fake_numpy = _FakeNumpy("numpy")
fake_scipy_special = types.ModuleType("scipy.special")
fake_scipy_special.logit = _fake_logit
fake_scipy_special.expit = _fake_expit

_register("numpy", fake_numpy)
_register("scipy.special", fake_scipy_special)

from unified_planning.engines.results import PlanGenerationResultStatus


class DummyActionInstance:
    def __init__(self, name="test_action", parameters=None):
        self.action = SimpleNamespace(name=name)
        self.actual_parameters = tuple(parameters or ("arg_a", "arg_b"))

    def __str__(self):
        params = ",".join(map(str, self.actual_parameters))
        return f"{self.action.name}({params})"


class _FakeFluentNode:
    def __init__(self, name: str = "dummy_fluent"):
        self._name = name
        self.args = []

    def fluent(self):
        return SimpleNamespace(name=self._name)


class TestPolicyCPP(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake"})
    def test_next_action_returns_policy_action(self):
        from viplan.policies.policy_cpp import PolicyCPP
        from viplan.policies.policy_interface import PolicyAction, PolicyObservation

        dummy_action_instance = DummyActionInstance()

        planner_mock = MagicMock()
        planner_mock.solve.return_value = SimpleNamespace(
            status=PlanGenerationResultStatus.SOLVED_SATISFICING,
            plan=SimpleNamespace(
                root_node=SimpleNamespace(
                    action_instance=dummy_action_instance,
                    children=[],
                )
            ),
        )

        with patch("viplan.policies.policy_cpp.OpenAIVQA", return_value=MagicMock()):
            with patch("viplan.policies.policy_cpp.create_up_problem", return_value=MagicMock()):
                with patch("viplan.policies.policy_cpp.to_contingent_problem", return_value=MagicMock()):
                    with patch(
                        "viplan.policies.policy_cpp.get_mapping_from_compiled_actions_to_original_actions",
                        return_value=lambda action: action,
                    ):
                        with patch(
                            "viplan.policies.policy_cpp.get_all_grounded_predicates_for_objects",
                            return_value=[_FakeFluentNode()],
                        ):
                            with patch(
                                "viplan.policies.policy_cpp.UPSequentialSimulator",
                                return_value=MagicMock(),
                            ):
                                with patch(
                                    "viplan.policies.policy_cpp.environment.get_environment",
                                    side_effect=_get_environment_stub,
                                ):
                                    with patch(
                                        "viplan.policies.policy_cpp.OneshotPlanner",
                                        return_value=planner_mock,
                                    ):
                                        policy = PolicyCPP(
                                            predicate_language={"dummy_fluent": "template"},
                                            domain_file="domain.pddl",
                                            problem_file="problem.pddl",
                                            model_name="gpt-test",
                                            base_prompt="prompt",
                                            tasks_logger=logging.getLogger("policy_cpp_test"),
                                            use_unknown_token=False,
                                        )

                                        observation = PolicyObservation(
                                            image=MagicMock(name="image"),
                                            problem=MagicMock(),
                                            predicate_language={},
                                            previous_actions=[],
                                        )

                                        action = policy.next_action(observation, log_extra={})

                                        self.assertIsInstance(action, PolicyAction)
                                        self.assertEqual(action.name, dummy_action_instance.action.name)
                                        self.assertEqual(
                                            action.parameters,
                                            [str(p) for p in dummy_action_instance.actual_parameters],
                                        )
