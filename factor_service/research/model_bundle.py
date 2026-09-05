"""Write a root model bundle and one independently verifiable rolling series."""
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile


def require_window_artifact(manifest):
    """Rolling models have exactly one supported layout: a separate series."""
    reference = manifest.get('walk_forward_artifact')
    if not isinstance(reference, dict) or not reference:
        raise ValueError('滚动模型缺少独立序列包清单，不再支持内嵌窗口格式')
    if (reference.get('artifact_kind') != 'walk_forward_series'
            or not re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256') or ''))
            or type(reference.get('size_bytes')) is not int
            or reference['size_bytes'] <= 0):
        raise ValueError('滚动模型独立序列包清单无效')
    return reference


def package_model_bundle(work_dir, manifest, members, *, has_windows=False):
    root = Path(work_dir)
    bundle = root / 'qlib_experiment.tar.gz'
    series = root / 'walk_forward_series.tar.gz'
    if has_windows:
        with tarfile.open(series, 'w:gz') as archive:
            archive.add(root / 'walk_forward', arcname='walk_forward')
        digest = sha256()
        with series.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        manifest['walk_forward_artifact'] = {
            'artifact_kind': 'walk_forward_series', 'sha256': digest.hexdigest(),
            'size_bytes': series.stat().st_size,
        }
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    with tarfile.open(bundle, 'w:gz') as archive:
        for name, path in members:
            archive.add(path, arcname=name)
    return bundle, series
