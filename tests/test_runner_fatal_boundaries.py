"""Inject faults into real runner handlers/finalizers without importing GPU models.

The explicit catalogue keeps a regression from disappearing when a handler is
removed. Only the failing operation is substituted; handlers and finalizers
are loaded from source. Structured outcomes, mock call traces and reference-release state verify
ordinary behavior against reviewed static fixtures. Cleanup completion and
mutation tests prevent an entered-only sentinel from masking failures. Existing
full-method subprocess tests cover Qwen, LTX25 and MiniMax integration.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = (
    ("lightx2v/models/runners/autoregressive_runner.py", "AutoregressiveRunnerMixin.run_warmup", 0, True),
    ("lightx2v/models/runners/autoregressive_runner.py", "AutoregressiveRunnerMixin.streaming_decode", 0, True),
    ("lightx2v/models/runners/autoregressive_runner.py", "AutoregressiveRunnerMixin.streaming_decode", 1, False),
    ("lightx2v/models/runners/base_runner.py", "BaseRunner.check_stop", 0, False),
    ("lightx2v/models/runners/cosmos3/cosmos3_runner.py", "Cosmos3Runner._run_action_forward_multichunk_pipeline", 0, True),
    ("lightx2v/models/runners/default_runner.py", "DefaultRunner.switch_lora", 0, False),
    ("lightx2v/models/runners/default_runner.py", "DefaultRunner.switch_lora_multi", 0, False),
    ("lightx2v/models/runners/hunyuan_image3/cuda_graph.py", "HunyuanImage3ARCudaGraphController._capture", 0, False),
    ("lightx2v/models/runners/hunyuan_image3/cuda_graph.py", "HunyuanImage3ARCudaGraphController._capture", 1, False),
    ("lightx2v/models/runners/hunyuan_image3/cuda_graph.py", "HunyuanImage3ARCudaGraphController._capture", 2, False),
    ("lightx2v/models/runners/hunyuan_image3/flashinfer_autotune.py", "FlashInferAutotuneController._load_distributed_cache", 0, False),
    ("lightx2v/models/runners/hunyuan_image3/flashinfer_autotune.py", "FlashInferAutotuneController._save_distributed_cache", 0, False),
    ("lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py", "HunyuanImage3Runner.close", 0, True),
    ("lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py", "HunyuanImage3Runner.close", 1, True),
    ("lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py", "HunyuanImage3Runner.run_pipeline", 0, True),
    ("lightx2v/models/runners/lingbot_video/lingbot_video_runner.py", "LingBotVideoRunner._run_warmup", 0, True),
    ("lightx2v/models/runners/ltx2/ltx25_runner.py", "LTX25Runner.run_pipeline", 0, False),
    ("lightx2v/models/runners/ltx2/ltx25_runner.py", "LTX25Runner.run_pipeline", 1, True),
    ("lightx2v/models/runners/ltx2/ltx2_runner.py", "LTX2Runner._run_warmup", 0, True),
    ("lightx2v/models/runners/ltx2/ltx2_runner.py", "LTX2Runner._run_warmup", 1, True),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_main", 0, True),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_main", 1, True),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_main", 2, False),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_main", 3, True),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_main", 4, False),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_warmup", 0, True),
    ("lightx2v/models/runners/minimax_h3/minimax_h3_runner.py", "MiniMaxH3Runner.run_warmup", 1, False),
    ("lightx2v/models/runners/neopp/neopp_runner.py", "NeoppRunner.run_pipeline", 0, True),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner._run_pipeline_local", 0, True),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner._run_warmup", 0, True),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner._run_warmup", 1, True),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner.load_vae", 0, False),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner.run", 0, True),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner.run_pipeline", 0, True),
    ("lightx2v/models/runners/qwen_image/qwen_image_runner.py", "QwenImageRunner.run_warmup", 0, True),
    ("lightx2v/models/runners/relayvsr/relayvsr_runner.py", "RelayVSRRunner.restore_frames", 0, True),
    ("lightx2v/models/runners/relayvsr/relayvsr_runner.py", "RelayVSRRunner.run_video", 0, True),
    ("lightx2v/models/runners/relayvsr/relayvsr_runner.py", "RelayVSRRunner.run_video", 1, False),
    ("lightx2v/models/runners/relayvsr/relayvsr_runner.py", "RelayVSRRunner.run_warmup", 0, True),
    ("lightx2v/models/runners/relayvsr/video_io.py", "AsyncVideoWriter._write", 0, False),
    ("lightx2v/models/runners/relayvsr/video_io.py", "AsyncVideoWriter.close", 0, True),
    ("lightx2v/models/runners/relayvsr/video_io.py", "AsyncVideoWriter.submit", 0, False),
    ("lightx2v/models/runners/seedvr/seedvr_runner.py", "SeedVRRunner._run_input_encoder_local_sr", 0, False),
    ("lightx2v/models/runners/seedvr/seedvr_runner.py", "SeedVRRunner.run_pipeline", 0, False),
    ("lightx2v/models/runners/seedvr/seedvr_runner.py", "SeedVRRunner.run_vae_decoder", 0, True),
    ("lightx2v/models/runners/seedvr/seedvr_runner.py", "SeedVRRunner.run_vae_encoder", 0, True),
    ("lightx2v/models/runners/swiftvr/swiftvr_runner.py", "SwiftVRRunner.copy_frames_to_cpu", 0, False),
    ("lightx2v/models/runners/swiftvr/swiftvr_runner.py", "SwiftVRRunner.run_pipeline", 0, True),
    ("lightx2v/models/runners/swiftvr/swiftvr_runner.py", "SwiftVRRunner.run_video_pipeline", 1, True),
    ("lightx2v/models/runners/swiftvr/swiftvr_runner.py", "SwiftVRRunner.run_warmup", 0, True),
    ("lightx2v/models/runners/swiftvr/swiftvr_runner.py", "SwiftVRRunner.write_video_frames", 0, False),
    ("lightx2v/models/runners/wan/wan_animate2_runner.py", "WanAnimate2Runner._vae_encode", 0, True),
    ("lightx2v/models/runners/wan/wan_animate2_runner.py", "WanAnimate2Runner.end_run", 0, True),
    ("lightx2v/models/runners/wan/wan_animate2_runner.py", "WanAnimate2Runner.init_run_segment", 0, False),
    ("lightx2v/models/runners/wan/wan_animate2_runner.py", "WanAnimate2Runner.run_main", 0, False),
    ("lightx2v/models/runners/wan/wan_animate2_runner.py", "WanAnimate2Runner.run_main", 1, False),
    ("lightx2v/models/runners/wan/wan_animate2_runner.py", "WanAnimate2Runner.run_segment", 0, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioARRunner.prefill_reference_kv", 0, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioARRunner.run_main", 0, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioARRunner.run_main", 1, False),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioARRunner.run_main", 2, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioARRunner.run_main_with_fixed_audio", 0, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioRunner.clear_warmup_state", 0, False),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioRunner.clear_warmup_state", 1, False),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioRunner.run_main", 0, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioRunner.run_main", 1, False),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioRunner.run_warmup", 0, True),
    ("lightx2v/models/runners/wan/wan_audio_runner.py", "WanAudioRunner.run_warmup", 1, False),
    ("lightx2v/models/runners/wan/wan_infinitetalk_runner.py", "InfiniteTalkRunner.clear_warmup_state", 0, True),
    ("lightx2v/models/runners/wan/wan_infinitetalk_runner.py", "InfiniteTalkRunner.process_images_after_vae_decoder", 0, True),
    ("lightx2v/models/runners/wan/wan_infinitetalk_runner.py", "InfiniteTalkRunner.run_pipeline", 0, True),
    ("lightx2v/models/runners/wan/wan_infinitetalk_runner.py", "InfiniteTalkRunner.run_pipeline", 1, True),
    ("lightx2v/models/runners/wan/wan_infinitetalk_runner.py", "InfiniteTalkRunner.run_warmup", 1, True),
    ("lightx2v/models/runners/wan/wan_infinitetalk_runner.py", "InfiniteTalkRunner.run_warmup", 2, False),
    ("lightx2v/models/runners/wan/wan_lingbot_fast_runner.py", "LingbotFastRunner.end_run", 0, True),
    ("lightx2v/models/runners/wan/wan_lingbot_fast_runner.py", "LingbotFastRunner.get_vae_encoder_output", 0, True),
    ("lightx2v/models/runners/wan/wan_lingbot_va_runner.py", "_StreamingVAEEncoder.encode_chunk", 0, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "Wan22DenseRunner.get_vae_encoder_output", 0, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner._run_pipeline_local", 0, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner._run_warmup", 0, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner._run_warmup", 1, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner._run_warmup", 2, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner.run_pipeline", 0, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner.run_segment", 0, True),
    ("lightx2v/models/runners/wan/wan_runner.py", "WanRunner.run_warmup", 0, True),
    ("lightx2v/models/runners/worldmirror/worldmirror_runner.py", "WorldMirrorRunner.run_pipeline", 1, False),
)


class RunnerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ROOT / "tests/data/runner_ordinary_results.jsonl"
        rows = [json.loads(line) for line in fixtures.read_text().splitlines()]
        cls.expected_ordinary = {row["key"]: row["expected"] for row in rows}
        keys = {f"{filename}:{scope}:{index}" for filename, scope, index, _ in BOUNDARIES}
        if len(rows) != len(cls.expected_ordinary) or set(cls.expected_ordinary) != keys:
            raise AssertionError("ordinary expectations must cover every boundary exactly once")

    def run_boundary(self, filename, scope, index, message, mutation=None):
        code = f"""
import ast, builtins, copy, importlib.util, json, re
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
node = ast.parse((root / {filename!r}).read_text())
for name in {scope!r}.split('.'):
    node = next(n for n in node.body if getattr(n, 'name', None) == name)
tries = []

def is_guard(n):
    return isinstance(n, ast.With) and any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == 'fatal_device_error_guard'
        for item in n.items
    )

# A try/finally with a guard body is one boundary, like the original handler.
covered = {{id(n.body[0]) for n in ast.walk(node) if isinstance(n, ast.Try)
           and not n.handlers and len(n.body) == 1 and is_guard(n.body[0])}}
def collect(n):
    if n is not node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    if isinstance(n, ast.Try) or (is_guard(n) and id(n) not in covered):
        tries.append(n)
    for child in ast.iter_child_nodes(n):
        collect(child)
collect(node)
boundary = copy.deepcopy(sorted(tries, key=lambda n: n.lineno)[{index}])
if isinstance(boundary, ast.Try) and not boundary.handlers and len(boundary.body) == 1 and is_guard(boundary.body[0]):
    boundary.body[0].body = ast.parse('raise injected').body
    boundary.body[0].items = [item for item in boundary.body[0].items if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id in ('fatal_device_error_guard', 'suppress')]
else:
    boundary.body = ast.parse('raise injected').body
if isinstance(boundary, ast.With):
    # Keep real guard/suppression semantics; isolate other device contexts.
    boundary.items = [item for item in boundary.items if
        isinstance(item.context_expr, ast.Call) and
        isinstance(item.context_expr.func, ast.Name) and
        item.context_expr.func.id in ('fatal_device_error_guard', 'suppress')]
    boundary = ast.Try(body=[boundary], handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Raise()])], orelse=[], finalbody=[])
mutation = {mutation!r}
if mutation == 'harness_crash':
    raise RuntimeError('injected child harness crash')
if mutation == 'skip_cleanup':
    boundary.finalbody = [ast.Pass()]
elif mutation == 'raise_cleanup':
    boundary.finalbody = ast.parse("raise ValueError('cleanup regression')").body
elif mutation == 'change_exception':
    boundary.body = ast.parse("raise ValueError('unexpected replacement')").body
elif mutation == 'change_return':
    for handler in boundary.handlers:
        for part in ast.walk(handler):
            if isinstance(part, ast.Return):
                part.value = ast.Constant('unexpected return')
if mutation == 'handler_late_fatal_exit':
    handler = boundary.handlers[0]
    guard = handler.body.pop(0)
    assert isinstance(guard, ast.Expr) and isinstance(guard.value, ast.Call)
    assert guard.value.func.id == 'terminate_on_fatal_device_error'
    handler.body.append(guard)
if boundary.finalbody:
    boundary.finalbody.insert(0, ast.parse("record_event(['cleanup', 'entered'])").body[0])
    boundary.finalbody.append(ast.parse("record_event(['cleanup', 'completed'])").body[0])
loop = ast.For(target=ast.Name(id='_once', ctx=ast.Store()), iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()), body=[boundary], orelse=[])
fn = ast.FunctionDef(name='exercise', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[loop], decorator_list=[])
events = []
def record_event(event):
    events.append(event)
    # Publish before os._exit bypasses final reporting.
    print('EVENT ' + json.dumps(event), flush=True)

def normalize(value):
    if isinstance(value, MagicMock):
        return {{'mock': value._extract_mock_name()}}
    if isinstance(value, BaseException):
        return {{'exception': type(value).__name__, 'message': str(value)}}
    if isinstance(value, (list, tuple)):
        return [normalize(x) for x in value]
    if isinstance(value, dict):
        return {{str(k): normalize(v) for k, v in value.items()}}
    if isinstance(value, str):
        return re.sub(r" id='[0-9]+'", '', value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return re.sub(r'0x[0-9a-f]+', '<address>', repr(value))
original_mock_call = MagicMock._execute_mock_call

def observed_mock_call(self, *args, **kwargs):
    # Preserve calls before os._exit without changing mock names/repr contracts.
    print('CALL ' + json.dumps(self._extract_mock_name()), flush=True)
    return original_mock_call(self, *args, **kwargs)

MagicMock._execute_mock_call = observed_mock_call

class Namespace(dict):
    def __missing__(self, key):
        value = MagicMock(name=key)
        self[key] = value
        return value
ns = Namespace(vars(builtins))
ns.update(injected=RuntimeError({message!r}), events=events, record_event=record_event, suppress=suppress,
          terminate_on_fatal_device_error=errors.terminate_on_fatal_device_error,
          fatal_device_error_guard=errors.fatal_device_error_guard)
if mutation == 'late_fatal_exit':
    ns['terminate_on_fatal_device_error'] = lambda error: None
    original_terminate = errors.terminate_on_fatal_device_error
    errors.terminate_on_fatal_device_error = lambda error: None
# Causal cache cleanup returns counts, not an unconstrained mock to unpack.
ns['self'].vae_decoder.clear_causal_memory.return_value = (0, 0)
ns['self'].vae_encoder.clear_causal_memory.return_value = (0, 0)
ns['self']._probe_video_torchcodec.return_value = (10, 24.0, [])
ns['super'] = lambda *a: ns['_parent_runner']
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<boundary>', 'exec'), ns)
class DeviceContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        record_event(['context_exit', None if exc_type is None else exc_type.__name__])
try:
    with DeviceContext():
        returned = ns['exercise']()
    outcome = {{'kind': 'return', 'value': normalize(returned)}}
except BaseException as observed:
    outcome = {{'kind': 'raise', 'type': type(observed).__name__, 'message': str(observed)}}
if mutation == 'late_fatal_exit':
    original_terminate(ns['injected'])
calls = {{}}
for name, value in ns.items():
    if isinstance(value, MagicMock) and value.mock_calls:
        # Calls are normalized structurally; no dynamic mock ids enter fixtures.
        calls[name] = [[call[0], normalize(call[1]), normalize(call[2])] for call in value.mock_calls]
state = {{}}
def mock_state(value, path):
    for name, field in vars(value).items():
        if not name.startswith(('_mock', '_spec')) and name not in (
            'called', 'call_count', 'call_args', 'call_args_list', 'method_calls', 'mock_calls',
            'return_value', 'side_effect'):
            state[path + '.' + name] = normalize(field)
    for name, child in value._mock_children.items():
        if isinstance(child, MagicMock):
            mock_state(child, path + '.' + name)
        else:
            state[path + '.' + name] = 'deleted'
for name, value in ns.items():
    if isinstance(value, MagicMock):
        mock_state(value, name)
print('OBSERVATION ' + json.dumps({{'outcome': outcome, 'events': events, 'calls': calls, 'state': state}}, sort_keys=True))
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def event_stream(self, result):
        events = []
        for line in result.stdout.splitlines():
            if not line.startswith("EVENT "):
                continue
            try:
                event = json.loads(line.removeprefix("EVENT "))
            except json.JSONDecodeError as error:
                self.fail(f"invalid event stream: {error}")
            self.assertIsInstance(event, list)
            self.assertEqual(len(event), 2)
            self.assertIn(event[0], ("cleanup", "context_exit"))
            events.append(event)
        return events

    def call_stream(self, result):
        calls = []
        for line in result.stdout.splitlines():
            if line.startswith("CALL "):
                name = json.loads(line.removeprefix("CALL "))
                self.assertIsInstance(name, str)
                calls.append(name)
        return calls

    def observation(self, result):
        # Execution validity is separate from whether a mutation changes behavior.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("LightX2V fatal inference failure", result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("OBSERVATION ")]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        try:
            observed = json.loads(lines[0].removeprefix("OBSERVATION "))
        except json.JSONDecodeError as error:
            self.fail(f"invalid observation: {error}")
        self.assertIsInstance(observed, dict)
        self.assertEqual(set(observed), {"outcome", "events", "calls", "state"})
        self.assertEqual(self.event_stream(result), observed["events"])
        return observed

    def assert_fatal_process_exit(self, result):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("LightX2V fatal inference failure", result.stderr)
        self.assertNotIn("OBSERVATION ", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def assert_fatal_before_cleanup(self, result):
        self.assert_fatal_process_exit(result)
        self.assertEqual(self.event_stream(result), [], "cleanup/context exit preceded fatal termination")
        self.assertFalse(self.call_stream(result), "handler call preceded fatal termination")

    def expected(self, filename, scope, index):
        key = f"{filename}:{scope}:{index}"
        return self.expected_ordinary[key]

    def assert_ordinary(self, filename, scope, index, result):
        self.assertEqual(self.observation(result), self.expected(filename, scope, index))

    def test_fatal_exception_exits_before_cleanup_or_context_exit(self):
        for filename, scope, index, has_cleanup in BOUNDARIES:
            with self.subTest(file=filename, scope=scope, boundary=index):
                result = self.run_boundary(filename, scope, index, "CUDA error: an illegal memory access was encountered")
                self.assert_fatal_before_cleanup(result)

    def test_ordinary_exception_type_result_and_actual_cleanup(self):
        for filename, scope, index, has_cleanup in BOUNDARIES:
            with self.subTest(file=filename, scope=scope, boundary=index):
                result = self.run_boundary(filename, scope, index, "ordinary failure")
                self.assert_ordinary(filename, scope, index, result)
                if has_cleanup:
                    events = self.observation(result)["events"]
                    self.assertIn(["cleanup", "entered"], events)
                    self.assertIn(["cleanup", "completed"], events)

    def assert_mutation_detected(self, filename, scope, index, result):
        observed = self.observation(result)  # A crash/missing output fails here.
        self.assertNotEqual(observed, self.expected(filename, scope, index))

    def test_late_fatal_exit_is_rejected(self):
        for filename, scope, index, has_cleanup in BOUNDARIES:
            with self.subTest(file=filename, scope=scope, boundary=index):
                result = self.run_boundary(filename, scope, index, "CUDA error: an illegal memory access was encountered", mutation="late_fatal_exit")
                # Require a valid fatal exit and valid persisted events BEFORE
                # expecting the ordering assertion to reject this regression.
                self.assert_fatal_process_exit(result)
                self.assertTrue(self.event_stream(result))
                with self.assertRaisesRegex(AssertionError, "cleanup/context exit preceded"):
                    self.assert_fatal_before_cleanup(result)

    def test_handler_fallback_before_fatal_exit_is_rejected(self):
        cases = (
            ("SeedVRRunner._run_input_encoder_local_sr", "self._read_video_segment_torchcodec"),
            ("SeedVRRunner.run_pipeline", "self._probe_video_torchcodec"),
        )
        filename = "lightx2v/models/runners/seedvr/seedvr_runner.py"
        for scope, fallback in cases:
            with self.subTest(scope=scope):
                result = self.run_boundary(filename, scope, 0, "CUDA error: an illegal memory access was encountered", mutation="handler_late_fatal_exit")
                self.assert_fatal_process_exit(result)
                self.assertEqual(self.event_stream(result), [])
                self.assertIn(fallback, self.call_stream(result))
                with self.assertRaisesRegex(AssertionError, "handler call preceded"):
                    self.assert_fatal_before_cleanup(result)
                # Moving the guard does not change the ordinary fallback contract.
                ordinary = self.run_boundary(filename, scope, 0, "ordinary failure", mutation="handler_late_fatal_exit")
                self.assert_ordinary(filename, scope, 0, ordinary)

    def test_unchanged_observation_is_not_a_detected_mutation(self):
        filename = "lightx2v/models/runners/neopp/neopp_runner.py"
        scope = "NeoppRunner.run_pipeline"
        result = self.run_boundary(filename, scope, 0, "ordinary failure")
        self.assert_ordinary(filename, scope, 0, result)
        with self.assertRaises(AssertionError):
            self.assert_mutation_detected(filename, scope, 0, result)

    def test_child_crash_is_not_a_detected_mutation(self):
        filename = "lightx2v/models/runners/neopp/neopp_runner.py"
        scope = "NeoppRunner.run_pipeline"
        result = self.run_boundary(filename, scope, 0, "ordinary failure", mutation="harness_crash")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("injected child harness crash", result.stderr)
        with self.assertRaises(AssertionError):
            self.assert_mutation_detected(filename, scope, 0, result)

    def test_missing_or_malformed_observation_is_not_a_detected_mutation(self):
        filename = "lightx2v/models/runners/neopp/neopp_runner.py"
        scope = "NeoppRunner.run_pipeline"
        for stdout in ("", "OBSERVATION invalid-json\n", "OBSERVATION {}\n"):
            with self.subTest(stdout=stdout):
                result = subprocess.CompletedProcess([], 0, stdout, "")
                with self.assertRaises(AssertionError):
                    self.assert_mutation_detected(filename, scope, 0, result)

    def test_missing_cleanup_is_rejected(self):
        for filename, scope, index, has_cleanup in BOUNDARIES:
            if not has_cleanup:
                continue
            with self.subTest(file=filename, scope=scope, boundary=index):
                result = self.run_boundary(filename, scope, index, "ordinary failure", mutation="skip_cleanup")
                self.assert_mutation_detected(filename, scope, index, result)

    def test_cleanup_that_raises_is_rejected(self):
        for filename, scope, index, has_cleanup in BOUNDARIES:
            if not has_cleanup:
                continue
            with self.subTest(file=filename, scope=scope, boundary=index):
                result = self.run_boundary(filename, scope, index, "ordinary failure", mutation="raise_cleanup")
                self.assert_mutation_detected(filename, scope, index, result)

    def test_changed_exception_and_return_are_rejected(self):
        filename = "lightx2v/models/runners/default_runner.py"
        scope = "DefaultRunner.switch_lora"
        result = self.run_boundary(filename, scope, 0, "ordinary failure", mutation="change_return")
        self.assert_mutation_detected(filename, scope, 0, result)
        filename = "lightx2v/models/runners/neopp/neopp_runner.py"
        scope = "NeoppRunner.run_pipeline"
        result = self.run_boundary(filename, scope, 0, "ordinary failure", mutation="change_exception")
        self.assert_mutation_detected(filename, scope, 0, result)


if __name__ == "__main__":
    unittest.main()
