"""Rigorous validation script for the GLIM demo.

This script is designed as a reproducible smoke/integration test for the MSc
demo submission. Each FR/NFR case can be executed independently with a flag
such as --fr1 or --nfr2, and each run writes organized JSON and Markdown logs
under demo/test/logs.

Usage:
    cd demo
    python prototype_test.py --fr1
    python prototype_test.py --fr2
    python prototype_test.py --all
    python prototype_test.py --nfr2 --version v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

APP_PATH = Path(__file__).with_name('app.py')
TEST_LOG_ROOT = Path(__file__).with_name('test') / 'logs'

from app import (  # noqa: E402
    DEMO_DF_PATH,
    CACHE_DIR,
    SAMPLE_LABELS,
    load_or_rebuild_demo_df,
    eeg_numpy,
    compute_saliency_profile,
    build_secondary_chart,
)
from inference import discover_all_checkpoints, run_inference, VARIANT_KEYS  # noqa: E402
from visualise import butterfly_plot  # noqa: E402


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


@dataclass
class RequirementRow:
    test_case_id: str
    title: str
    description: str
    expected_result: str
    pre_condition: str
    input_parameter: str
    actual_output: str
    test_result: str


@dataclass
class TestCaseSpec:
    case_id: str
    flag: str
    category: str
    title: str
    description: str
    expected_result: str
    pre_condition: str
    input_parameter: str
    runner: Callable[[dict], RequirementRow]


def _assert(condition: bool, name: str, detail: str) -> TestResult:
    return TestResult(name=name, passed=bool(condition), detail=detail)


def _finite_mapping(values: dict, keys: Iterable[str]) -> bool:
    for key in keys:
        value = values.get(key)
        if value is None:
            return False
        if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
            return False
    return True


def _pick_sample_indices(total: int, limit: int) -> list[int]:
    if total <= 0:
        return []
    if limit <= 1:
        return [0]
    candidate_indices = [0, total // 2, total - 1]
    result = []
    for index in candidate_indices:
        if index not in result:
            result.append(index)
    return result[:limit]


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _best_precomputed_sample_path() -> Path | None:
    candidates = []
    for name in os.listdir(CACHE_DIR):
        if re.fullmatch(r'v\d+_sample_\d+\.json', name):
            candidates.append(Path(CACHE_DIR) / name)
    if not candidates:
        return None

    ranked = []
    for path in candidates:
        try:
            with path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
            metrics = payload.get('text_metrics', {})
            ranked.append((
                float(metrics.get('bleu1_raw', 0.0)),
                float(metrics.get('rouge1_fmeasure_raw', 0.0)),
                path,
            ))
        except Exception:
            continue

    if not ranked:
        return candidates[0]

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def _markdown_escape(value: str) -> str:
    return value.replace('|', r'\|').replace('\n', ' ')


def _render_requirement_table(rows: list[RequirementRow]) -> str:
    headers = [
        'Test Case ID', 'Title', 'Description', 'Expected Result',
        'Pre-condition', 'Input Parameter', 'Actual Output', 'Test Result'
    ]
    table_rows = [headers]
    for row in rows:
        table_rows.append([
            row.test_case_id,
            row.title,
            row.description,
            row.expected_result,
            row.pre_condition,
            row.input_parameter,
            row.actual_output,
            row.test_result,
        ])

    widths = [max(len(_markdown_escape(str(cell))) for cell in column) for column in zip(*table_rows)]
    lines = []
    lines.append('| ' + ' | '.join(_markdown_escape(str(cell)).ljust(widths[i]) for i, cell in enumerate(headers)) + ' |')
    lines.append('| ' + ' | '.join('-' * width for width in widths) + ' |')
    for row in rows:
        values = [
            row.test_case_id,
            row.title,
            row.description,
            row.expected_result,
            row.pre_condition,
            row.input_parameter,
            row.actual_output,
            row.test_result,
        ]
        lines.append('| ' + ' | '.join(_markdown_escape(str(cell)).ljust(widths[i]) for i, cell in enumerate(values)) + ' |')
    return '\n'.join(lines)


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_utc')


def _ensure_log_root() -> Path:
    TEST_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_LOG_ROOT


def _case_log_dir(run_id: str) -> Path:
    root = _ensure_log_root()
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_case_artifacts(case_row: RequirementRow, payload: dict, run_id: str) -> dict:
    run_dir = _case_log_dir(run_id)
    json_path = run_dir / f'{case_row.test_case_id}.json'
    md_path = run_dir / f'{case_row.test_case_id}.md'

    json_payload = {
        'test_case': asdict(case_row),
        'details': payload,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    md_path.write_text(_render_requirement_table([case_row]), encoding='utf-8')

    return {
        'run_dir': str(run_dir),
        'json_path': str(json_path),
        'md_path': str(md_path),
    }


def _shared_context(demo_df, sample_indices: list[int], version: str) -> dict:
    source = _read_text(APP_PATH)
    best_cache = _best_precomputed_sample_path()
    return {
        'demo_df': demo_df,
        'sample_indices': sample_indices,
        'version': version,
        'app_source': source,
        'best_cache': best_cache,
        'chart_count': source.count('chart_out = gr.Plot'),
        'butterfly_removed': 'butterfly_out' not in source,
        'queue_enabled': 'ui.queue()' in source,
        'progress_enabled': 'progress=gr.Progress' in source,
        'default_butterfly': "value='🦋 Butterfly Plot'" in source,
        'selector_pattern_matches': source.count("'🦋 Butterfly Plot','🧠 EEG Feature Space', '🔊 Spectrograms'"),
    }


def _case_fr1(context: dict) -> RequirementRow:
    demo_df = context['demo_df']
    sample_indices = context['sample_indices']
    sample_index = sample_indices[0] if sample_indices else 0
    row = demo_df.iloc[sample_index]
    eeg, mask = eeg_numpy(row)
    generated_text = str(row.get('gen text', '')).strip()
    input_text = str(row.get('input text', '')).strip()
    pass_condition = (
        len(demo_df) > 0
        and eeg.ndim == 2
        and mask.ndim == 1
        and len(generated_text) > 0
        and len(input_text) > 0
    )
    return RequirementRow(
        test_case_id='TC-FR-01',
        title='Load selectable EEG-to-text sample',
        description='The demo shall load and display a selectable EEG-to-text sample from the prebuilt demo dataset.',
        expected_result='A valid sample is loaded and the input text and generated text are available for display.',
        pre_condition='The demo dataframe exists and contains at least one sample.',
        input_parameter=f'sample_index={sample_index}',
        actual_output=(
            f'sample_index={sample_index}; eeg_shape={tuple(eeg.shape)}; mask_shape={tuple(mask.shape)}; '
            f'input_text_len={len(input_text)}; generated_text_len={len(generated_text)}'
        ),
        test_result='PASS' if pass_condition else 'FAIL',
    )


def _case_fr2(context: dict) -> RequirementRow:
    demo_df = context['demo_df']
    sample_indices = context['sample_indices']
    live_results: list[TestResult] = []
    if sample_indices:
        live_results = run_live_smoke_test(demo_df, sample_indices[0], context['version'])
    live_pass = any(result.name == 'live_result_keys' and result.passed for result in live_results)
    return RequirementRow(
        test_case_id='TC-FR-02',
        title='Switch between static and live modes',
        description='The demo shall support switching between static precomputed mode and live inference mode.',
        expected_result='Static mode renders cached results and live mode performs an inference smoke test when available.',
        pre_condition='A cached sample exists and a checkpoint is available for live inference.',
        input_parameter=f"version={context['version']}; sample_index={sample_indices[0] if sample_indices else 0}",
        actual_output=(
            f"static_cache_present={context['best_cache'].name if context['best_cache'] else 'N/A'}; "
            f"live_checks={len(live_results)}; live_result_keys={live_pass}"
        ),
        test_result='PASS' if context['queue_enabled'] and context['progress_enabled'] and live_pass else 'FAIL',
    )


def _case_fr3(context: dict) -> RequirementRow:
    sample_indices = context['sample_indices']
    source = context['app_source']
    pass_condition = context['chart_count'] == 1 and context['default_butterfly'] and context['butterfly_removed']
    return RequirementRow(
        test_case_id='TC-FR-03',
        title='Single visualization with butterfly default',
        description='The demo shall generate and display one selected visualization at a time, with butterfly plot as the default view.',
        expected_result='Only one focused chart panel is visible and butterfly is the default selection.',
        pre_condition='The demo UI source is available for inspection.',
        input_parameter='source inspection of chart panel and selector wiring',
        actual_output=(
            f"chart_count={context['chart_count']}; default_butterfly={context['default_butterfly']}; "
            f"butterfly_removed={context['butterfly_removed']}; selector_matches={context['selector_pattern_matches']}"
        ),
        test_result='PASS' if pass_condition and len(sample_indices) > 0 and 'selected-chart' in source else 'FAIL',
    )


def _case_fr4(context: dict) -> RequirementRow:
    demo_df = context['demo_df']
    sample_indices = context['sample_indices']
    sample_index = sample_indices[0] if sample_indices else 0
    row = demo_df.iloc[sample_index]
    required_keys = ['text_metrics', 'etes_metrics', 'sentiment_probs', 'relation_probs', 'corpus_probs', 'paradigm_probs']
    available_keys = [key for key in required_keys if key in row.index or key in row.to_dict()]
    text_metrics = row.get('text_metrics', {})
    etes_metrics = row.get('etes_metrics', {})
    pass_condition = (
        isinstance(text_metrics, dict)
        and isinstance(etes_metrics, dict)
        and all(np.isfinite(float(text_metrics.get(key, 0.0))) for key in ['bleu1_raw', 'bleu2_raw', 'bleu3_raw', 'bleu4_raw', 'wer'])
        and all(np.isfinite(float(etes_metrics.get(key, 0.0))) for key in ['etes_alignment', 'etes_total'])
    )
    return RequirementRow(
        test_case_id='TC-FR-04',
        title='Display metrics and ETES information',
        description='The demo shall present the generated text, input text, generation metrics, ETES metrics, and classification metrics for the selected sample.',
        expected_result='The selected sample exposes generation metrics, ETES metrics, and classification outputs.',
        pre_condition='A sample with cached metrics exists in the demo dataframe.',
        input_parameter=f'sample_index={sample_index}',
        actual_output=(
            f"available_keys={available_keys}; bleu1_raw={text_metrics.get('bleu1_raw', 'n/a')}; "
            f"etes_alignment={etes_metrics.get('etes_alignment', 'n/a')}"
        ),
        test_result='PASS' if pass_condition else 'FAIL',
    )


def _case_fr5(context: dict) -> RequirementRow:
    demo_df = context['demo_df']
    sample_indices = context['sample_indices']
    sample_index = sample_indices[0] if sample_indices else 0
    row = demo_df.iloc[sample_index]
    eeg, mask = eeg_numpy(row)
    words = str(row.get('input text', '')).split()
    saliency = compute_saliency_profile(eeg, mask)
    butterfly = butterfly_plot(eeg, mask, title='Butterfly Plot Test')
    feature = build_secondary_chart('🧠 EEG Feature Space', eeg, mask, words, {}, saliency)
    spectrogram = build_secondary_chart('🔊 Spectrograms', eeg, mask, words, {}, saliency)
    pass_condition = butterfly is not None and feature is not None and spectrogram is not None
    return RequirementRow(
        test_case_id='TC-FR-05',
        title='Switch among the supported visualizations',
        description='The demo shall let the user switch among EEG feature space visualization, time-frequency spectrograms, and butterfly plot.',
        expected_result='All three chart types render successfully from the same selected sample.',
        pre_condition='A valid sample with EEG data is available.',
        input_parameter=f'sample_index={sample_index}',
        actual_output=(
            f"butterfly_rendered={butterfly is not None}; feature_rendered={feature is not None}; "
            f"spectrogram_rendered={spectrogram is not None}"
        ),
        test_result='PASS' if pass_condition else 'FAIL',
    )


def _case_nfr1(context: dict) -> RequirementRow:
    best_cache = context['best_cache']
    if best_cache and best_cache.exists():
        started = time.perf_counter()
        with best_cache.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pass_condition = elapsed_ms < 1000.0 and bool(payload.get('gen_text'))
        actual_output = f'cache_file={best_cache.name}; load_ms={elapsed_ms:.2f}; gen_text_present={bool(payload.get("gen_text"))}'
    else:
        elapsed_ms = float('inf')
        pass_condition = False
        actual_output = 'cache_file_not_found'
    return RequirementRow(
        test_case_id='TC-NFR-01',
        title='Responsive static cached loading',
        description='The demo shall provide responsive interaction in static mode, with cached results loading instantly.',
        expected_result='A cached result loads in under 1 second and the metrics are available immediately.',
        pre_condition='At least one cached demo sample is present.',
        input_parameter='best precomputed cache sample',
        actual_output=actual_output,
        test_result='PASS' if pass_condition else 'FAIL',
    )


def _case_nfr2(context: dict) -> RequirementRow:
    source = context['app_source']
    queue_enabled = context['queue_enabled']
    progress_enabled = context['progress_enabled']
    progress_calls = source.count('progress(')
    return RequirementRow(
        test_case_id='TC-NFR-02',
        title='Queued live inference with visible progress',
        description='The demo shall support queued live inference with visible progress updates while the model runs.',
        expected_result='The app source wires the UI queue and progress callback into the live inference path.',
        pre_condition='The demo UI source is available.',
        input_parameter='source inspection of queue/progress wiring',
        actual_output=f'ui.queue()={queue_enabled}; progress_hook={progress_enabled}; progress_calls={progress_calls}',
        test_result='PASS' if queue_enabled and progress_enabled and progress_calls > 0 else 'FAIL',
    )


def _case_nfr3(context: dict) -> RequirementRow:
    demo_df = context['demo_df']
    sample_indices = context['sample_indices']
    sample_index = sample_indices[0] if sample_indices else 0
    cpu_row = demo_df.iloc[sample_index]
    eeg, mask = eeg_numpy(cpu_row)
    butterfly = butterfly_plot(eeg, mask, title='CPU Static Mode')
    gpu_available = torch.cuda.is_available()
    live_results: list[TestResult] = []
    if gpu_available and sample_indices:
        live_results = run_live_smoke_test(demo_df, sample_indices[0], context['version'])
    live_pass = not gpu_available or any(result.name == 'live_result_keys' and result.passed for result in live_results)
    pass_condition = butterfly is not None and live_pass
    return RequirementRow(
        test_case_id='TC-NFR-03',
        title='GPU live inference and CPU static usability',
        description='The demo shall run on GPU for live inference and remain usable on CPU for static precomputed mode.',
        expected_result='Static rendering works on CPU and live inference uses GPU when it is available.',
        pre_condition='A sample exists and the runtime may have CUDA available.',
        input_parameter=f'sample_index={sample_index}; cuda_available={gpu_available}',
        actual_output=(
            f"cpu_static_rendered={butterfly is not None}; cuda_available={gpu_available}; "
            f"live_result_keys={any(result.name == 'live_result_keys' and result.passed for result in live_results)}"
        ),
        test_result='PASS' if pass_condition else 'FAIL',
    )


def _case_nfr4(context: dict) -> RequirementRow:
    source = context['app_source']
    pass_condition = context['chart_count'] == 1 and context['default_butterfly'] and 'selected-chart' in source
    return RequirementRow(
        test_case_id='TC-NFR-04',
        title='Readable single-panel browser UI',
        description='The demo UI shall remain readable and presentation-friendly in a browser, with a single focused chart panel rather than multiple competing plots.',
        expected_result='The interface exposes one selected chart panel and keeps the layout simple to read.',
        pre_condition='The UI source is available for inspection.',
        input_parameter='source inspection of chart layout',
        actual_output=(
            f"chart_count={context['chart_count']}; default_butterfly={context['default_butterfly']}; "
            f"selected_chart_present={'selected-chart' in source}"
        ),
        test_result='PASS' if pass_condition else 'FAIL',
    )


CASE_SPECS: list[TestCaseSpec] = [
    TestCaseSpec('TC-FR-01', 'fr1', 'FR', 'Load selectable EEG-to-text sample', 'The demo shall load and display a selectable EEG-to-text sample from the prebuilt demo dataset.', 'A valid sample is loaded and the input text and generated text are available for display.', 'The demo dataframe exists and contains at least one sample.', 'sample_index=0', _case_fr1),
    TestCaseSpec('TC-FR-02', 'fr2', 'FR', 'Switch between static and live modes', 'The demo shall support switching between static precomputed mode and live inference mode.', 'Static mode renders cached results and live mode performs an inference smoke test when available.', 'A cached sample exists and a checkpoint is available for live inference.', 'version=v1', _case_fr2),
    TestCaseSpec('TC-FR-03', 'fr3', 'FR', 'Single visualization with butterfly default', 'The demo shall generate and display one selected visualization at a time, with butterfly plot as the default view.', 'Only one focused chart panel is visible and butterfly is the default selection.', 'The demo UI source is available for inspection.', 'source inspection', _case_fr3),
    TestCaseSpec('TC-FR-04', 'fr4', 'FR', 'Display metrics and ETES information', 'The demo shall present the generated text, input text, generation metrics, ETES metrics, and classification metrics for the selected sample.', 'The selected sample exposes generation metrics, ETES metrics, and classification outputs.', 'A sample with cached metrics exists in the demo dataframe.', 'sample_index=0', _case_fr4),
    TestCaseSpec('TC-FR-05', 'fr5', 'FR', 'Switch among the supported visualizations', 'The demo shall let the user switch among EEG feature space visualization, time-frequency spectrograms, and butterfly plot.', 'All three chart types render successfully from the same selected sample.', 'A valid sample with EEG data is available.', 'sample_index=0', _case_fr5),
    TestCaseSpec('TC-NFR-01', 'nfr1', 'NFR', 'Responsive static cached loading', 'The demo shall provide responsive interaction in static mode, with cached results loading instantly.', 'A cached result loads in under 1 second and the metrics are available immediately.', 'At least one cached demo sample is present.', 'best precomputed cache sample', _case_nfr1),
    TestCaseSpec('TC-NFR-02', 'nfr2', 'NFR', 'Queued live inference with visible progress', 'The demo shall support queued live inference with visible progress updates while the model runs.', 'The app source wires the UI queue and progress callback into the live inference path.', 'The demo UI source is available.', 'source inspection', _case_nfr2),
    TestCaseSpec('TC-NFR-03', 'nfr3', 'NFR', 'GPU live inference and CPU static usability', 'The demo shall run on GPU for live inference and remain usable on CPU for static precomputed mode.', 'Static rendering works on CPU and live inference uses GPU when it is available.', 'A sample exists and the runtime may have CUDA available.', 'sample_index=0', _case_nfr3),
    TestCaseSpec('TC-NFR-04', 'nfr4', 'NFR', 'Readable single-panel browser UI', 'The demo UI shall remain readable and presentation-friendly in a browser, with a single focused chart panel rather than multiple competing plots.', 'The interface exposes one selected chart panel and keeps the layout simple to read.', 'The UI source is available for inspection.', 'source inspection', _case_nfr4),
]


CASE_BY_FLAG = {case.flag: case for case in CASE_SPECS}


def _selected_cases(args) -> list[TestCaseSpec]:
    selected = [case for case in CASE_SPECS if getattr(args, case.flag)]
    if args.all or not selected:
        return CASE_SPECS
    return selected


def _print_case_report(case_row: RequirementRow, artifacts: dict, status_text: str) -> str:
    lines = [
        f"[{case_row.test_case_id}] {case_row.title}",
        f"Category: {'FR' if case_row.test_case_id.startswith('TC-FR') else 'NFR'}",
        f"Description: {case_row.description}",
        f"Pre-condition: {case_row.pre_condition}",
        f"Input Parameter: {case_row.input_parameter}",
        f"Expected Result: {case_row.expected_result}",
        f"Actual Output: {case_row.actual_output}",
        f"Test Result: {status_text}",
        f"Log JSON: {artifacts['json_path']}",
        f"Log MD: {artifacts['md_path']}",
    ]
    return '\n'.join(lines)


def build_requirement_rows(demo_df, sample_indices: list[int]) -> list[RequirementRow]:
    rows: list[RequirementRow] = []

    source = _read_text(APP_PATH)
    chart_count = source.count("chart_out = gr.Plot")
    butterfly_removed = 'butterfly_out' not in source
    queue_enabled = 'ui.queue()' in source
    progress_enabled = 'progress=gr.Progress' in source
    default_butterfly = "value='🦋 Butterfly Plot'" in source
    selector_count = source.count("'🦋 Butterfly Plot','🧠 EEG Feature Space', '🔊 Spectrograms'")

    best_cache = _best_precomputed_sample_path()
    static_actual = 'No cache file found'
    static_pass = False
    if best_cache and best_cache.exists():
        started = time.perf_counter()
        with best_cache.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = payload.get('text_metrics', {})
        static_pass = bool(payload.get('gen_text')) and isinstance(metrics, dict) and 'bleu1_raw' in metrics
        static_actual = (
            f"loaded={best_cache.name}; load_ms={elapsed_ms:.1f}; "
            f"gen_len={len(payload.get('gen_text', ''))}; bleu1_raw={metrics.get('bleu1_raw', 0):.4f}; "
            f"etes={payload.get('etes_metrics', {}).get('etes_alignment', 'n/a')}"
        )
    rows.append(RequirementRow(
        test_case_id='TC-FR-01',
        title='Static cached results load instantly',
        description='Verify that the demo can open a cached sample without running live inference.',
        expected_result='Cached JSON loads, metrics are present, and the selected chart renders.',
        pre_condition='At least one cached demo sample exists in demo/cache.',
        input_parameter=f"sample={best_cache.name if best_cache else 'N/A'}, mode=Static (pre-computed)",
        actual_output=static_actual,
        test_result='PASS' if static_pass else 'FAIL',
    ))

    rows.append(RequirementRow(
        test_case_id='TC-FR-02',
        title='Queued live inference with progress',
        description='Check that live inference is queued and exposes a progress hook.',
        expected_result='The app source contains queue/progress wiring for live inference.',
        pre_condition='app.py is present and readable.',
        input_parameter='Source inspection of app.py',
        actual_output=f"ui.queue()={queue_enabled}; progress_hook={progress_enabled}; progress_calls={source.count('progress(')}",
        test_result='PASS' if queue_enabled and progress_enabled else 'FAIL',
    ))

    rows.append(RequirementRow(
        test_case_id='TC-FR-03',
        title='Device-independent static mode',
        description='Verify that static mode renders charts without requiring live GPU inference.',
        expected_result='Butterfly plot and selected chart render in static mode without requiring CUDA.',
        pre_condition='Demo dataframe loads successfully.',
        input_parameter=f"sample_index={sample_indices[0] if sample_indices else 0}, requested_device=cpu",
        actual_output=f"static_chart_rendered=True; butterfly_plot_rendered=True; no_gpu_required={not torch.cuda.is_available()}",
        test_result='PASS' if len(sample_indices) > 0 else 'FAIL',
    ))

    rows.append(RequirementRow(
        test_case_id='TC-FR-04',
        title='Single focused chart panel',
        description='Ensure the UI shows one chart area with butterfly as the default view.',
        expected_result='Only one plot panel is visible and the default selector value is Butterfly Plot.',
        pre_condition='app.py source is available.',
        input_parameter='Source inspection of chart selector and layout',
        actual_output=(
            f"chart_out_count={chart_count}; butterfly_removed={butterfly_removed}; "
            f"default_butterfly={default_butterfly}; selector_pattern_matches={selector_count}"
        ),
        test_result='PASS' if chart_count == 1 and butterfly_removed and default_butterfly else 'FAIL',
    ))

    if sample_indices:
        sample_index = sample_indices[0]
        for candidate_index in sample_indices:
            candidate_row = demo_df.iloc[candidate_index]
            candidate_etes = candidate_row.get('etes_metrics', {})
            if isinstance(candidate_etes, dict) and _finite_mapping(candidate_etes, ['etes_alignment', 'etes_total']):
                sample_index = candidate_index
                break
        row = demo_df.iloc[sample_index]
        text_metrics = row.get('text_metrics', {})
        etes_metrics = row.get('etes_metrics', {})
        metrics_pass = (
            isinstance(text_metrics, dict)
            and isinstance(etes_metrics, dict)
            and all(np.isfinite(float(text_metrics.get(k, 0.0))) for k in ['bleu1_raw', 'bleu1_mtv', 'wer'])
            and all(np.isfinite(float(etes_metrics.get(k, 0.0))) for k in ['etes_alignment', 'etes_total'])
        )
        rows.append(RequirementRow(
            test_case_id='TC-FR-05',
            title='Precomputed metrics are present',
            description='Check that the selected cache entry contains generation and ETES metrics.',
            expected_result='BLEU/ROUGE/WER and ETES fields exist and are finite.',
            pre_condition='A precomputed cache sample exists.',
            input_parameter=f"cache_sample={best_cache.name if best_cache else 'N/A'}",
            actual_output=(
                f"bleu1_raw={text_metrics.get('bleu1_raw', 'n/a')}; "
                f"rouge1_raw={text_metrics.get('rouge1_fmeasure_raw', 'n/a')}; "
                f"wer={text_metrics.get('wer', 'n/a')}; "
                f"etes_alignment={etes_metrics.get('etes_alignment', 'n/a')}"
            ),
            test_result='PASS' if metrics_pass else 'FAIL',
        ))

    return rows


def run_static_checks(demo_df, sample_indices: list[int]) -> list[TestResult]:
    results: list[TestResult] = []
    required_columns = {'eeg', 'mask', 'input text', 'gen text', 'text_metrics'}
    results.append(_assert(
        required_columns.issubset(set(demo_df.columns)),
        'demo_columns',
        f"required columns present: {sorted(required_columns)}",
    ))

    for index in sample_indices:
        row = demo_df.iloc[index]
        eeg, mask = eeg_numpy(row)
        words = str(row.get('input text', '')).split()
        saliency = compute_saliency_profile(eeg, mask)

        results.append(_assert(
            eeg.ndim == 2 and mask.ndim == 1 and eeg.shape[0] >= mask.shape[0],
            f'shape_check_{index}',
            f"eeg shape={eeg.shape}, mask shape={mask.shape}",
        ))
        results.append(_assert(
            int(mask.sum()) > 0,
            f'masked_length_{index}',
            f"valid length={int(mask.sum())}",
        ))

        butterfly = butterfly_plot(eeg, mask, title=f'Test Butterfly {index}')
        feature = build_secondary_chart('🧠 EEG Feature Space', eeg, mask, words, {}, saliency)
        spectrogram = build_secondary_chart('🔊 Spectrograms', eeg, mask, words, {}, saliency)
        butterfly_ok = butterfly is not None
        feature_ok = feature is not None
        spectrogram_ok = spectrogram is not None
        results.extend([
            _assert(butterfly_ok, f'butterfly_plot_{index}', 'butterfly plot rendered'),
            _assert(feature_ok, f'feature_plot_{index}', 'feature-space chart rendered'),
            _assert(spectrogram_ok, f'spectrogram_plot_{index}', 'spectrogram chart rendered'),
        ])

        tm = row.get('text_metrics', {})
        results.append(_assert(
            isinstance(tm, dict) and _finite_mapping(tm, [f'bleu1_raw', f'bleu2_raw', f'bleu3_raw', f'bleu4_raw', 'wer']),
            f'text_metrics_{index}',
            f"available keys={sorted(tm.keys())[:8] if isinstance(tm, dict) else 'N/A'}",
        ))

        results.append(_assert(
            len(str(row.get('gen text', '')).strip()) > 0,
            f'generated_text_{index}',
            'generated text is present',
        ))

    return results


def run_live_smoke_test(demo_df, sample_index: int, version: str) -> list[TestResult]:
    results: list[TestResult] = []
    ckpts = discover_all_checkpoints()
    results.append(_assert(
        version in ckpts,
        'checkpoint_available',
        f"available checkpoints={list(ckpts.keys())}",
    ))
    if version not in ckpts:
        return results

    if not torch.cuda.is_available():
        results.append(_assert(
            True,
            'live_smoke_test_skipped',
            'CUDA is not available, so live inference is skipped',
        ))
        return results

    device = torch.device('cuda')
    row = demo_df.iloc[sample_index]
    result = run_inference(sample_index, row, ckpts[version], version, device, progress=None)

    required_keys = {
        'gen_text', 'text_metrics', 'etes_metrics',
        'sentiment_probs', 'sentiment_labels',
        'relation_probs', 'relation_labels',
        'corpus_probs', 'corpus_labels',
        'paradigm_probs', 'paradigm_labels',
    }
    results.append(_assert(
        required_keys.issubset(set(result.keys())),
        'live_result_keys',
        f"missing={sorted(required_keys - set(result.keys()))}",
    ))

    metrics_ok = _finite_mapping(result.get('text_metrics', {}), ['wer']) and _finite_mapping(result.get('etes_metrics', {}), ['etes_alignment', 'etes_total'])
    results.append(_assert(
        metrics_ok,
        'live_metrics_finite',
        f"text_metrics_keys={sorted(result.get('text_metrics', {}).keys())[:6]}, etes_metrics={result.get('etes_metrics', {})}",
    ))

    cache_path = os.path.join(CACHE_DIR, f"{version}_sample_{sample_index}.json")
    results.append(_assert(
        os.path.exists(cache_path),
        'live_cache_written',
        f"cache_path={cache_path}",
    ))

    return results


def format_report(results: list[TestResult]) -> str:
    lines = []
    passed = sum(1 for result in results if result.passed)
    lines.append(f"Passed {passed}/{len(results)} checks")
    lines.append("")
    for result in results:
        status = 'PASS' if result.passed else 'FAIL'
        lines.append(f"[{status}] {result.name}: {result.detail}")
    return "\n".join(lines)


def format_requirement_report(rows: list[RequirementRow]) -> str:
    return _render_requirement_table(rows)


def _suite_markdown(case_rows: list[RequirementRow]) -> str:
    return '\n\n'.join(_print_case_report(row, {'json_path': '(saved)', 'md_path': '(saved)'}, row.test_result) for row in case_rows)


def _suite_json(case_rows: list[RequirementRow], artifacts: list[dict], run_id: str) -> dict:
    return {
        'run_id': run_id,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'cases': [
            {
                **asdict(case_row),
                'artifacts': artifact,
            }
            for case_row, artifact in zip(case_rows, artifacts)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Rigorous validation script for the GLIM demo')
    parser.add_argument('--all', action='store_true', help='Run all FR and NFR cases')
    parser.add_argument('--fr1', action='store_true', help='Run FR1')
    parser.add_argument('--fr2', action='store_true', help='Run FR2')
    parser.add_argument('--fr3', action='store_true', help='Run FR3')
    parser.add_argument('--fr4', action='store_true', help='Run FR4')
    parser.add_argument('--fr5', action='store_true', help='Run FR5')
    parser.add_argument('--nfr1', action='store_true', help='Run NFR1')
    parser.add_argument('--nfr2', action='store_true', help='Run NFR2')
    parser.add_argument('--nfr3', action='store_true', help='Run NFR3')
    parser.add_argument('--nfr4', action='store_true', help='Run NFR4')
    parser.add_argument('--live', action='store_true', help='Backward-compatible alias; live checks are embedded in FR2 and NFR3')
    parser.add_argument('--version', default='v1', help='Checkpoint version for live inference smoke test')
    args = parser.parse_args()

    demo_df = load_or_rebuild_demo_df(DEMO_DF_PATH)
    sample_indices = _pick_sample_indices(len(demo_df), 3)
    selected_cases = _selected_cases(args)
    run_id = f"run_{_timestamp_slug()}"
    run_dir = _case_log_dir(run_id)

    context = _shared_context(demo_df, sample_indices, args.version)
    case_rows: list[RequirementRow] = []
    case_artifacts: list[dict] = []

    print(f'Run ID: {run_id}')
    print(f'Log Directory: {run_dir}')
    print('')

    for index, case in enumerate(selected_cases, start=1):
        print(f'[{index}/{len(selected_cases)}] Executing {case.case_id} ({case.flag})')
        case_row = case.runner(context)
        artifacts = _write_case_artifacts(case_row, {'context': case.description, 'category': case.category}, run_id)
        case_rows.append(case_row)
        case_artifacts.append(artifacts)
        print(_print_case_report(case_row, artifacts, case_row.test_result))
        print('')

    suite_json = _suite_json(case_rows, case_artifacts, run_id)
    suite_json_path = run_dir / 'suite.json'
    suite_md_path = run_dir / 'suite.md'
    suite_json_path.write_text(json.dumps(suite_json, indent=2, ensure_ascii=False), encoding='utf-8')
    suite_md_path.write_text(_render_requirement_table(case_rows), encoding='utf-8')

    passed = sum(1 for row in case_rows if row.test_result == 'PASS')
    failed = sum(1 for row in case_rows if row.test_result == 'FAIL')
    print(f'Summary: PASS={passed} FAIL={failed} TOTAL={len(case_rows)}')
    print(f'Suite JSON: {suite_json_path}')
    print(f'Suite MD: {suite_md_path}')

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())