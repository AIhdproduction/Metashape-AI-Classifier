"""The state behind the window, free of NiceGUI. All rules about what may change live here."""

import logging
import math
import shutil
import threading
import time
from pathlib import Path

import psutil
from pydantic import ValidationError

from msam3 import backends, gui_options, gui_progress, user_classes
from msam3.app import i18n
from msam3.cache import atomic_json, file_hash
from msam3.config import (
    REPO_ROOT,
    Classes,
    Settings,
    load_classes,
    load_settings,
    merge_settings,
)
from msam3.gui_progress import RunView, read_json
from msam3.segmentation import model_files

LOG = logging.getLogger(__name__)
STORAGE_INTERVAL_SECONDS = 10
INTEGER_OPTIONS = (
    "sam_prompt_batch_size",
    "sam_min_views",
    "flurseg_batch_size",
    "flurseg_min_views",
    "mask_interior_px",
)


class ActionError(Exception):
    """An action that is not possible; ``key`` selects the translated reason."""

    def __init__(self, key: str, **values):
        super().__init__(key)
        self.key, self.values = key, values


def launcher_process(session_data: dict):
    """The Metashape process that waits for this session, or None when it is gone."""
    pid = session_data.get("launcher_pid")
    if pid is None:
        return None
    try:
        process = psutil.Process(int(pid))
        # A PID created after the session belongs to some other program.
        if process.create_time() > float(session_data.get("created_at", math.inf)) + 2:
            return None
        return process
    except (psutil.Error, ValueError):
        return None


class Controller:
    def __init__(
        self,
        session: Path | None = None,
        *,
        settings: Settings | None = None,
        sessions_root: Path | None = None,
        classes_file: Path | None = None,
        language_file: Path | None = None,
    ):
        self.lock = threading.RLock()
        self.base = settings or load_settings()
        self.sessions_root = sessions_root or (
            session.parent if session else REPO_ROOT / "work/gui-sessions"
        )
        self.classes_file = classes_file
        self.language_file = language_file
        self.t = i18n.Translator(i18n.load_language(language_file))
        self.standard_classes = load_classes()
        self.revision = 0  # Grows whenever the page has to be rebuilt.
        self.cancelled_here = False
        self.storage: dict = {}
        self._storage_at = -math.inf
        self.log = ""
        self.weights_state = ""  # v1 | unknown | checking | ""
        self._weights_cache: dict = {}
        self.status: dict[str, backends.ModelStatus] = {}
        self._reload_classes()
        # The live session of the launcher; a recent project is only looked at.
        self.own_session = session
        self.session: Path | None = None
        self.viewing = False
        self.recent = gui_progress.recent_sessions(self.sessions_root) if not session else []
        self._open(session)
        self.recheck_models()

    # ----------------------------------------------------------------------------- session

    def _open(self, session: Path | None, viewing: bool = False) -> None:
        self.session, self.viewing = session, viewing
        data = read_json(session / "session.json") if session else None
        if session and not data:
            raise FileNotFoundError(f"No session.json in {session}")
        self.data = data or {}
        self.project = str(self.data.get("project", ""))
        self.image_sizes = [tuple(size) for size in self.data.get("image_sizes", [])]
        self.image_count = int(self.data.get("image_count", 0))
        self.options = self._options_of(session)
        self.launcher = launcher_process(self.data)
        self.history = {}
        if session and not viewing:
            root, project = self.sessions_root, self.project
            self.history = {
                "flurseg": gui_options.historical_seconds_per_window(root, project),
                "sam3": gui_options.historical_seconds_per_image(
                    root, self.image_sizes, project, None
                ),
            }
        self.job = {}
        self.run = RunView("ready")
        self.storage, self._storage_at = {}, -math.inf
        self.log = ""
        self.revision += 1
        self.poll()

    def _options_of(self, session: Path | None) -> gui_options.RunOptions:
        """A started session shows what it runs with, also in a window that was opened later."""
        default = gui_options.RunOptions.from_settings(self.base)
        chosen = read_json(session / "options.json") if session else None
        if not chosen:
            return default
        merged = merge_settings(
            self.base.model_dump(mode="json"), chosen.get("settings_override", {})
        )
        # Sessions from before the model choice were always SAM 3.
        merged["segmentation"] = {"backend": chosen.get("model", {}).get("backend", "sam3")}
        try:
            return gui_options.RunOptions.from_settings(Settings.model_validate(merged))
        except ValueError as error:
            LOG.warning("The options of %s cannot be shown: %s", session, error)
            return default

    @property
    def project_name(self) -> str:
        return Path(self.project.replace("\\", "/")).stem

    @property
    def standalone(self) -> bool:
        return self.own_session is None

    def open_recent(self, session: Path) -> None:
        if not self.standalone:
            raise ActionError("lock.running")
        if not (session / "session.json").is_file():
            raise ActionError("error.session_gone")
        with self.lock:
            self._open(session, viewing=True)

    def close_recent(self) -> None:
        if self.standalone:
            with self.lock:
                self.recent = gui_progress.recent_sessions(self.sessions_root)
                self._open(None)

    def _launcher_alive(self) -> bool:
        if "launcher_pid" not in self.data:
            # A viewed session without this information is taken as it was written.
            return not self.viewing
        try:
            return self.launcher is not None and self.launcher.is_running()
        except psutil.Error:
            return False

    def poll(self) -> None:
        """Read the files again. Called once per second and after every action."""
        with self.lock:
            self._read_session()
        self._measure_storage()

    def _read_session(self) -> None:
        session = self.session
        if session is None:
            self.run = RunView("ready")
        else:
            before = self.run
            run = gui_progress.read_run(session, self._launcher_alive())
            log = gui_progress.read_log(session)
            job = self.job or read_json(session / "job.json", {})
            self.run, self.log, self.job = run, log, job
            changed = (before.state, before.phase, before.cancel_requested) != (
                run.state,
                run.phase,
                run.cancel_requested,
            )
            changed |= before.worker_running != run.worker_running
            if changed or before.result != run.result:
                self.revision += 1
                self._storage_at = -math.inf

    def _measure_storage(self) -> None:
        now = time.monotonic()
        if now - self._storage_at < STORAGE_INTERVAL_SECONDS:
            return
        self._storage_at = now
        storage = {}
        work = self.job.get("work_dir")
        if work and Path(work).is_dir():
            storage["work"] = gui_progress.directory_size(Path(work))
        existing = Path(work) if work else self.base.paths.work_dir
        while not existing.is_dir() and existing != existing.parent:
            existing = existing.parent
        try:
            storage["free"] = shutil.disk_usage(existing).free
        except OSError as error:
            LOG.info("Free disk space cannot be read: %s", error)
        self.storage = storage

    # -------------------------------------------------------------------------- what is allowed

    @property
    def options_lock(self) -> str | None:
        """Why the options of a run cannot be changed, as a text key."""
        if self.viewing:
            return "lock.viewing"
        if self.run.state == "running":
            return "lock.running"
        if self.session is not None and self.run.state != "ready":
            return "lock.finished"
        return None

    @property
    def classes_lock(self) -> str | None:
        return "lock.running" if self.run.state == "running" and not self.viewing else None

    @property
    def start_lock(self) -> str | None:
        if self.session is None:
            return "lock.standalone"
        if reason := self.options_lock:
            return reason
        status = self.status[self.options.backend]
        if not status.ready:
            return f"lock.model_{status.state}"
        if self.options.backend == "sam3" and self.classes_error:
            return "lock.classes"
        return None

    def _require(self, reason: str | None) -> None:
        if reason:
            raise ActionError(reason)

    # -------------------------------------------------------------------------------- options

    def select_model(self, backend: str) -> None:
        self._require(self.options_lock)
        if backend not in backends.MODELS:
            raise ActionError("error.invalid_setting", name="segmentation.backend")
        self.options.backend = backend
        self.revision += 1

    def select_profile(self, name: str) -> None:
        self._require(self.options_lock)
        if name not in gui_options.PROFILE_NAMES:
            raise ActionError("error.invalid_setting", name="profile")
        self.options.apply_profile(name)
        self.revision += 1

    def set_option(self, name: str, value) -> None:
        """Change one value. The page is not rebuilt, so a field keeps its focus while typing."""
        self._require(self.options_lock)
        if name == "backend" or name not in vars(self.options):
            raise ActionError("error.invalid_setting", name=name)
        if name in INTEGER_OPTIONS and isinstance(value, float) and value.is_integer():
            value = int(value)
        setattr(self.options, name, value)

    def auto_batch(self) -> tuple[int, float, float]:
        self._require(self.options_lock)
        memory = gui_options.gpu_memory_gib(self.base.gpu.device)
        if memory is None:
            raise ActionError("error.no_gpu_info")
        batch = gui_options.maximum_safe_batch(
            *memory, self.options.sam_image_scale, self.options.sam_tile_overlap
        )
        if batch is None:
            raise ActionError("error.no_safe_batch", free=f"{memory[0]:.1f}")
        self.options.sam_prompt_batch_size = batch
        self.revision += 1
        return batch, *memory

    def set_language(self, language: str) -> None:
        if language not in i18n.LANGUAGES:
            raise ActionError("error.invalid_setting", name="language")
        try:
            i18n.save_language(language, self.language_file)
        except OSError as error:
            raise ActionError("error.save", detail=str(error)) from error
        self.t = i18n.Translator(language)
        self.revision += 1

    def estimate(self) -> gui_options.Estimate | None:
        if self.session is None or self.viewing or not self.image_count:
            return None
        rate = self.history.get(self.options.backend)
        try:
            return self.options.estimate_from(rate, self.image_count, self.image_sizes)
        except (TypeError, ValueError, ArithmeticError):
            return None  # A field is empty or invalid at the moment; Start reports it.

    def run_settings(self) -> Settings:
        try:
            return self.options.settings(self.base)
        except ValidationError as error:
            name = ".".join(str(part) for part in error.errors()[0]["loc"])
            raise ActionError("error.invalid_setting", name=name) from error

    def imported_label(self) -> str:
        if self.viewing or self.run.state != "ready":
            if label := self.job.get("settings", {}).get("metashape", {}).get("imported_label"):
                return label
        return backends.MODELS[self.options.backend].imported_label(self.base)

    # --------------------------------------------------------------------------------- models

    def recheck_models(self) -> None:
        settings = self.base
        self.status = {key: model.status(settings) for key, model in backends.MODELS.items()}
        status = self.status["flurseg"]
        self.weights_state = "checking" if status.state in ("ready", "runtime") else ""
        self.revision += 1

    def identify_weights(self) -> None:
        """Compare with v1's checksum. It reads 1.2 GB, so it runs outside the UI thread."""
        if not self.weights_state:
            return
        path = self.status["flurseg"].path / "model.safetensors"
        try:
            stat = path.stat()
            if stat.st_size != model_files.FLURSEG_V1_BYTES:
                state = "unknown"
            else:
                key = (str(path), stat.st_mtime_ns)
                if key not in self._weights_cache:
                    self._weights_cache[key] = file_hash(path) == model_files.FLURSEG_V1_SHA256
                state = "v1" if self._weights_cache[key] else "unknown"
        except OSError as error:
            LOG.warning("FlurSeg weights cannot be read: %s", error)
            state = "unknown"
        if state != self.weights_state:
            self.weights_state = state
            self.revision += 1

    def download_command(self, backend: str) -> str | None:
        """The command that puts the files exactly at the configured path."""
        path = backends.MODELS[backend].path(self.base)
        standard = REPO_ROOT / ("models/flurseg/v1" if backend == "flurseg" else "models/sam3.pt")
        if path == standard:
            target = "models/flurseg" if backend == "flurseg" else "models"
        elif path.name == standard.name:
            target = f'"{path.parent}"'
        else:
            return None  # The download cannot rename its file; the page explains the copy.
        if backend == "flurseg":
            return model_files.FLURSEG_DOWNLOAD.replace("models/flurseg", target)
        return rf".venv\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir {target}"

    # -------------------------------------------------------------------------------- classes

    def _reload_classes(self) -> None:
        self.classes_error = ""
        try:
            self.edits = user_classes.read(self.classes_file)
            self.sam_classes = user_classes.apply(self.standard_classes, self.edits)
        except ValueError as error:
            LOG.warning("User classes ignored: %s", error)
            self.edits = user_classes.UserClasses()
            self.sam_classes = self.standard_classes
            self.classes_error = str(error)

    def run_classes(self, backend: str | None = None) -> Classes:
        backend = backend or self.options.backend
        return self.sam_classes if backend == "sam3" else backends.MODELS[backend].classes()

    def shown_class_count(self) -> int:
        if self.session is not None and self.job.get("classes"):
            if self.viewing or self.run.state != "ready":
                return len(self.job["classes"]["classes"])
        return len(self.run_classes().classes)

    def _change_classes(self, change) -> None:
        self._require(self.classes_lock)
        if self.classes_error:
            raise ActionError("error.classes_file")
        try:
            changed = change(self.standard_classes, self.edits)
            user_classes.apply(self.standard_classes, changed)
            user_classes.write(changed, self.classes_file)
        except user_classes.UserClassError as error:
            raise ActionError(f"class_error.{error.key}", **error.values) from error
        except (OSError, ValueError) as error:
            raise ActionError("error.save", detail=str(error)) from error
        self._reload_classes()
        self.revision += 1

    def save_class(self, name, prompts, color, code=None, original=None) -> None:
        self._change_classes(
            lambda standard, edits: user_classes.save_class(
                standard, edits, name, prompts, color, code, original
            )
        )

    def remove_class(self, name: str) -> None:
        self._change_classes(
            lambda standard, edits: user_classes.remove_class(standard, edits, name)
        )

    def set_class_enabled(self, name: str, enabled: bool) -> None:
        self._change_classes(
            lambda standard, edits: user_classes.set_enabled(standard, edits, name, enabled)
        )

    def set_prompts(self, name: str, prompts: list[str]) -> None:
        self._change_classes(
            lambda standard, edits: user_classes.set_prompts(standard, edits, name, prompts)
        )

    def reset_classes(self) -> None:
        self._require(self.classes_lock)
        try:
            user_classes.write(user_classes.UserClasses(), self.classes_file)
        except OSError as error:
            raise ActionError("error.save", detail=str(error)) from error
        self._reload_classes()
        self.revision += 1

    def inactive_helpers(self, backend: str | None = None) -> dict[str, list[str]]:
        backend = backend or self.options.backend
        try:
            settings = self.options.settings(self.base).model_copy(deep=True)
        except ValueError:
            settings = self.base.model_copy(deep=True)
        settings.segmentation.backend = backend
        return gui_options.inactive_helpers(settings, self.run_classes(backend))

    # ------------------------------------------------------------------------- start and cancel

    def start(self) -> None:
        with self.lock:
            self.poll()
            self._require(self.start_lock)
            # Files may have been removed since the window opened.
            model = backends.MODELS[self.options.backend]
            status = model.status(self.base)
            if not status.ready:
                self.status[model.key] = status
                self.revision += 1
                raise ActionError(f"lock.model_{status.state}")
            if (self.session / "options_cancelled.json").is_file():
                raise ActionError("lock.finished")
            settings = self.run_settings()
            estimate = self.estimate()
            payload = gui_options.options_payload(
                self.options,
                self.base,
                self.run_classes(),
                estimate.seconds if estimate else 0,
            )
            LOG.info(
                "Start with %s, profile %s, label %s",
                model.title,
                payload["profile"],
                model.imported_label(settings),
            )
            try:
                atomic_json(self.session / "options.json", payload)
            except OSError as error:
                raise ActionError("error.save", detail=str(error)) from error
            self.poll()

    def cancel_before_start(self) -> bool:
        with self.lock:
            if self.own_session is None or self.viewing:
                return False
            if (self.own_session / "options.json").is_file():
                return False
            atomic_json(self.own_session / "options_cancelled.json", {"cancelled_at": time.time()})
            self.cancelled_here = True
            self.poll()
            return True

    def stop_worker(self) -> None:
        """Also for a recent project: freeing the GPU changes nothing of that run's result."""
        if self.session is None or self.run.state != "error":
            raise ActionError("error.worker_gone")
        if not gui_progress.stop_worker(self.session):
            self.poll()
            raise ActionError("error.worker_gone")
        self.poll()
        if self.run.worker_running:
            raise ActionError("error.worker_alive")

    def cancel_run(self) -> None:
        if self.viewing or self.session is None or not self.run.can_cancel:
            raise ActionError("cancel.locked")
        if not gui_progress.request_cancel(self.session):
            raise ActionError("cancel.locked")
        self.poll()
