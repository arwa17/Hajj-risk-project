"""Private, durable, fail-closed before-send/after-response Gemini attempt journal.

No network is performed by this module. Every entry stays outside the Git tree.
A RECEIVED entry can be replayed offline. A STARTED/UNKNOWN entry is never
sent again automatically: delivery to Gemini may have occurred.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path


class AttemptJournalError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _save(path: Path, payload: dict) -> None:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    tmp = path.parent / ('.pending-' + uuid.uuid4().hex)
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(blob)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _append_event(path: Path, event: dict) -> None:
    """Separate immutable event stream for audit history, never overwrite events."""
    event_path = path.with_suffix('.events.jsonl')
    blob = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    fd = os.open(event_path, flags, 0o600)
    with os.fdopen(fd, 'ab', buffering=0) as out:
        out.write(blob)
        os.fsync(out.fileno())


class AttemptJournal:
    def __init__(self, directory: str | Path, *, allow_test_directory: bool = False):
        directory = Path(directory).expanduser()
        if not directory.is_absolute():
            raise AttemptJournalError('Journal directory must be an absolute path')
        home_private = Path.home() / 'CPCS683-V2-Private'
        if not allow_test_directory and not directory.is_relative_to(home_private):
            raise AttemptJournalError('Live journal must be under ~/CPCS683-V2-Private')
        # Reject symlink traversal before creating anything.
        for parent in (directory, *directory.parents):
            if parent.is_symlink():
                raise AttemptJournalError('Symlink in private journal path')
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory.is_dir():
            raise AttemptJournalError('Invalid journal path')
        if (directory.stat().st_mode & 0o077) != 0:
            raise AttemptJournalError('Private journal permissions must be owner-only (0700)')
        self.directory = directory

    def start_or_replay(self, scope: str, attempt_no: int, model_id: str,
                        exact_request_body: bytes, endpoint: str) -> tuple[Path, dict, bool]:
        if not scope or not isinstance(scope, str) or not 1 <= attempt_no <= 2:
            raise AttemptJournalError('Explicit invocation scope and bounded attempt required')
        identity = {'scope': scope, 'attempt': attempt_no, 'model_requested': model_id,
                    'request_sha256': sha256(exact_request_body),
                    'request_body_utf8': exact_request_body.decode('utf-8'),
                    'endpoint_without_credentials': endpoint}
        # Scope+attempt is the slot. Changed payload/model/settings MUST NOT get a new slot.
        key = sha256(json.dumps({'scope': scope, 'attempt': attempt_no},
                                sort_keys=True).encode('utf-8'))
        path = self.directory / (key + '.json')
        if path.is_file():
            if path.is_symlink():
                raise AttemptJournalError('Symlink in existing attempt record')
            try:
                existing = json.loads(path.read_text('utf-8'))
            except (ValueError, UnicodeError, OSError) as exc:
                raise AttemptJournalError('Corrupted existing attempt record') from exc
            if existing.get('identity') != identity:
                raise AttemptJournalError('Existing attempt identity is inconsistent')
            if existing.get('state') != 'RECEIVED':
                raise AttemptJournalError('UNKNOWN_DELIVERY: prior send not safely completed')
            try:
                raw = base64.b64decode(existing['raw_response_base64'], validate=True)
                if existing.get('raw_response_sha256') != sha256(raw):
                    raise ValueError('Bad raw response digest')
                if not isinstance(existing.get('http_status'), int):
                    raise ValueError('Invalid saved HTTP status')
            except (ValueError, KeyError, TypeError) as exc:
                raise AttemptJournalError('Incomplete/corrupt received response') from exc
            return path, existing, True
        if path.exists() or path.is_symlink():
            raise AttemptJournalError('Invalid attempt record path')
        fd = None
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'wb') as out:
                fd = None
                obj = {'identity': identity, 'state': 'STARTED', 'http_status': None,
                       'pre_send_utc': utc_now()}
                out.write(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf-8'))
                out.flush()
                os.fsync(out.fileno())
            dirfd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            _append_event(path, {'type': 'STARTED', 'utc': obj['pre_send_utc'],
                                 'identity': identity})
        except FileExistsError as exc:
            raise AttemptJournalError('Concurrent attempt or existing prior record') from exc
        finally:
            if fd is not None:
                os.close(fd)
        return path, obj, False

    def received(self, path: Path, entry: dict, raw: bytes, status: int, elapsed_ms: float) -> None:
        if entry.get('state') != 'STARTED' or path.is_symlink():
            raise AttemptJournalError('Invalid attempt response transition')
        next_entry = dict(entry, state='RECEIVED', http_status=status,
                          raw_response_base64=base64.b64encode(raw).decode('ascii'),
                          raw_response_sha256=sha256(raw),
                          received_utc=utc_now(), provider_elapsed_ms=elapsed_ms)
        _append_event(path, {'type': 'RECEIVED', 'utc': next_entry['received_utc'],
                             'http_status': status, 'provider_elapsed_ms': elapsed_ms,
                             'raw_response_base64': next_entry['raw_response_base64'],
                             'raw_response_sha256': next_entry['raw_response_sha256']})
        _save(path, next_entry)

    def unknown(self, path: Path, entry: dict, category: str) -> None:
        if entry.get('state') != 'STARTED' or path.is_symlink():
            raise AttemptJournalError('Invalid attempt unknown-delivery transition')
        when = utc_now()
        _append_event(path, {'type': 'UNKNOWN_DELIVERY', 'utc': when,
                             'error_category': category})
        _save(path, dict(entry, state='UNKNOWN_DELIVERY', error_category=category,
                         failed_utc=when))

    def annotate(self, path: Path, updates: dict) -> None:
        if path is None:
            return
        data = json.loads(path.read_text('utf-8'))
        if data.get('state') != 'RECEIVED':
            raise AttemptJournalError('Cannot annotate incomplete attempt')
        if 'identity' in updates or 'raw_response_base64' in updates:
            raise AttemptJournalError('Immutable attempt evidence cannot be overwritten')
        _append_event(path, {'type': 'ANNOTATION', 'utc': utc_now(), 'updates': updates})
        _save(path, {**data, **updates})
