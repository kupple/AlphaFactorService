from pathlib import Path
from hashlib import sha256
import json
import pickle
import tarfile

import pytest

from factor_service.model_research_repository import _inference_series_artifact, _job_row, ModelResearchConflict
from factor_service.research.inference import _download_window_bundle, _load_bundle, _load_model_for_trade_date
from factor_service.research.errors import PermanentJobError
from factor_service.research.model_bundle import package_model_bundle


@pytest.mark.parametrize('rolling', [False, True])
def test_packager_stores_window_bytes_once_and_freezes_series_digest(tmp_path, rolling):
    (tmp_path / 'model.pkl').write_bytes(pickle.dumps({'root': True}))
    window = tmp_path / 'walk_forward/window_0001'
    window.mkdir(parents=True)
    body = pickle.dumps({'window': 1})
    (window / 'model.pkl').write_bytes(body)
    manifest = {'model_id': 'model'}
    main, series = package_model_bundle(tmp_path, manifest, [
        ('manifest.json', tmp_path / 'manifest.json'), ('model.pkl', tmp_path / 'model.pkl'),
    ], has_windows=rolling)
    with tarfile.open(main) as archive:
        assert archive.getnames() == ['manifest.json', 'model.pkl']
        saved = json.load(archive.extractfile('manifest.json'))
    assert saved == manifest
    if rolling:
        assert manifest['walk_forward_artifact']['sha256'] == sha256(series.read_bytes()).hexdigest()
        assert manifest['walk_forward_artifact']['size_bytes'] == series.stat().st_size
        with tarfile.open(series) as archive:
            assert archive.extractfile('walk_forward/window_0001/model.pkl').read() == body
    else:
        assert not series.exists() and 'walk_forward_artifact' not in manifest


def test_series_identity_must_be_registered_and_match_root_manifest():
    ref = {'artifact_kind': 'walk_forward_series', 'sha256': 'a' * 64, 'size_bytes': 10}
    manifest = {'walk_forward': {'enabled': True}, 'walk_forward_artifact': ref}
    artifact = {**ref, 'artifact_id': 'series-id'}
    assert _inference_series_artifact({}, [artifact]) is None
    assert _inference_series_artifact(manifest, [artifact]) == {k: artifact[k] for k in ('artifact_id', 'sha256', 'size_bytes')}
    for rows in ([], [{**artifact, 'sha256': 'b' * 64}], [{**artifact, 'size_bytes': 11}]):
        with pytest.raises(ModelResearchConflict):
            _inference_series_artifact(manifest, rows)


def test_split_bundle_never_silently_falls_back_to_latest_model(tmp_path):
    manifest = {'walk_forward_artifact': {'artifact_kind': 'walk_forward_series', 'sha256': 'a' * 64, 'size_bytes': 10}}
    for source in ({}, {'walk_forward_artifact': {'artifact_id': 'wrong', 'sha256': 'b' * 64, 'size_bytes': 10}}):
        with pytest.raises(PermanentJobError, match='冻结模型清单'):
            _download_window_bundle(manifest, source, None, tmp_path)


def test_nonrolling_model_uses_root_model_without_downloading_a_series(tmp_path):
    model = {'static': True}
    manifest = {'walk_forward': {'enabled': False}}
    (tmp_path / 'model.pkl').write_bytes(pickle.dumps(model))
    bundle, _ = package_model_bundle(tmp_path, manifest, [
        ('model.pkl', tmp_path / 'model.pkl'), ('manifest.json', tmp_path / 'manifest.json'),
    ])
    loaded, metadata = _load_bundle(bundle)
    assert _load_model_for_trade_date(None, loaded, metadata, '2024-01-02') == (model, manifest, None)


@pytest.mark.parametrize('reference', [None, {}, {'artifact_kind': 'bundle'}, {'artifact_kind': 'walk_forward_series', 'sha256': 'a' * 64, 'size_bytes': True}])
def test_rolling_inference_rejects_missing_or_invalid_series_reference(tmp_path, reference):
    manifest = {'walk_forward': {'enabled': True}, 'walk_forward_artifact': reference}
    with pytest.raises(ModelResearchConflict, match='独立序列包清单'):
        _inference_series_artifact(manifest, [])
    with pytest.raises(PermanentJobError, match='独立序列包清单'):
        _download_window_bundle(manifest, {}, None, tmp_path)


@pytest.mark.parametrize('embedded', [False, True])
def test_old_rolling_bundle_is_rejected_before_model_deserialization(tmp_path, embedded):
    manifest = {'walk_forward': {'enabled': True}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'model.pkl').write_bytes(b'must not be deserialized')
    bundle = tmp_path / 'old.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for name in ('manifest.json', 'model.pkl'):
            archive.add(tmp_path / name, arcname=name)
        if embedded:
            archive.add(tmp_path / 'model.pkl', arcname='walk_forward/window_0001/model.pkl')
    with pytest.raises(PermanentJobError, match='独立序列包'):
        _load_bundle(bundle)


def test_old_success_status_is_100_without_rewriting_original_record():
    original = {'status': 'succeeded', 'progress_json': {'stage': 'completing', 'percent': 99, 'phase_seconds': {'uploading': 20}}}
    public = _job_row(original)
    assert public['progress_json']['stage'] == 'succeeded'
    assert public['progress_json']['percent'] == 100
    assert public['progress_json']['phase_seconds']['uploading'] == 20
    assert original['progress_json']['percent'] == 99
