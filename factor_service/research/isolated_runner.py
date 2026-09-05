from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal

from factor_service.model_artifacts import ModelArtifactStore
from factor_service.model_object_store import ModelObjectStore
from factor_service.model_research_repository import ModelResearchRepository
from factor_service.research.control import ResearchControl
from factor_service.research.config import load_settings
from factor_service.research.inference import DailyInferenceRunner
from factor_service.research.trainer import QlibTrainer, TrainingResult
from factor_service.research.job import CancellationToken
from factor_service.research.errors import error_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one model task in an isolated process")
    parser.add_argument("kind", choices=("train", "infer"))
    parser.add_argument("job_path", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job_path.read_text(encoding="utf-8"))
    settings = load_settings()
    if args.kind == "infer":
        result = DailyInferenceRunner(
            settings,
            ResearchControl(
                ModelResearchRepository(),
                ModelArtifactStore(settings.model_artifacts_root),
                ModelObjectStore(settings.model_object_store),
            ),
        ).run(job, args.work_dir)
    else:
        token = CancellationToken(timeout_seconds=(
            ((job.get('config_json') or {}).get('execution') or {}).get('max_runtime_minutes', 720) * 60))
        def cancel(signum, frame):
            token.cancel('父任务已停止，取消全部分布式子任务')
        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        def progress(stage, percent, details):
            data = (json.dumps(dict(stage=stage, percent=percent, details=details), default=str) + '\n').encode()
            descriptor = os.open(args.work_dir / 'isolated_progress.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, data)
            finally:
                os.close(descriptor)
        try:
            result = QlibTrainer(settings).train(job, args.work_dir, cancellation=token, progress=progress)
        except Exception as exc:
            _write_error(args.result_path, exc)
            raise
    _write_result(args.result_path, result)


def _write_result(path: Path, result: TrainingResult) -> None:
    payload = {
        "result": result.result,
        "artifacts": [[kind, str(artifact)] for kind, artifact in result.artifacts],
        "predictions_path": str(result.predictions_path),
    }
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_error(path: Path, exc: Exception) -> None:
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.error.tmp')
    temporary.write_text(json.dumps({'error': str(exc), **error_payload(exc)}, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
