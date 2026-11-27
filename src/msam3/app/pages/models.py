"""Models: both selectable, with their state on this PC and how to get the files."""

from nicegui import ui

from msam3 import backends
from msam3.app import components
from msam3.config import REPO_ROOT
from msam3.segmentation import model_files

FACTS = ("speed", "classes", "accuracy", "licence")
DOWNLOAD_PAGES = {
    "flurseg": f"{model_files.FLURSEG_URL}/tree/{model_files.FLURSEG_REVISION}/v1",
    "sam3": model_files.SAM_URL,
}


def render(screen) -> None:
    controller, t = screen.c, screen.t
    with ui.element("div").classes("mac-row mac-spread"):
        components.note(t("models.intro"), "mac-text")

        def recheck():
            screen.act(controller.recheck_models, success=t("models.rechecked"))

        components.button(t("action.recheck"), "refresh", recheck)
    with ui.element("div").classes("mac-two"):
        for key, model in backends.MODELS.items():
            _card(screen, key, model)


def _card(screen, key: str, model: backends.Model) -> None:
    controller, t = screen.c, screen.t
    status = controller.status[key]
    selected = key == controller.options.backend
    selectable = controller.session is not None and not controller.options_lock
    classes = "mac-card mac-model" + (" is-selected" if selected else "")
    with ui.element("div").classes(classes):
        with ui.element("div").classes("mac-row mac-spread"):
            ui.label(model.title).classes("mac-card-title")
            _state(screen, key, status)
        components.note(t(f"model.{key}.description"), "mac-text")
        with ui.element("div"):
            for fact in FACTS:
                with ui.element("div").classes("mac-fact"):
                    ui.label(t(f"models.fact.{fact}"))
                    ui.label(t(f"model.{key}.{fact}"))
        if not status.ready:
            if status.state == "missing":
                components.banner("warning", t(f"model.{key}.missing"))
            else:
                text = f"{t(f'lock.model_{status.state}')}\n{status.detail}"
                components.banner("error", text)
        components.note(t(f"model.{key}.target"))
        ui.label(str(status.path)).classes("mac-code")
        components.link(t(f"model.{key}.link"), DOWNLOAD_PAGES[key])
        command = controller.download_command(key)
        if command:
            full = f'cd "{REPO_ROOT}"\n{command}'
            with ui.element("div").classes("mac-row mac-spread"):
                components.note(t(f"model.{key}.command"))
                components.button(
                    t("action.copy"),
                    "content_copy",
                    lambda _, text=full: screen.copy(text),
                    "tool",
                )
            ui.label(full).classes("mac-code")
        else:
            components.note(t("models.manual_copy"))
        if key == "flurseg":
            components.note("Built with DINOv3")
        with ui.element("div").classes("mac-model-select"):
            if selected:
                with ui.element("div").classes("mac-state is-ready"):
                    ui.icon("radio_button_checked")
                    ui.label(t("models.selected"))
            else:
                reason = None
                if not selectable:
                    reason = t(controller.options_lock or "lock.standalone")
                components.button(
                    t("action.select_model", model=model.title),
                    "radio_button_unchecked",
                    lambda _, key=key: screen.act(controller.select_model, key),
                    "secondary",
                    reason,
                )


def _state(screen, key: str, status: backends.ModelStatus) -> None:
    t = screen.t
    icon = "check_circle" if status.ready else "error"
    look = "is-ready" if status.ready else "is-problem"
    with ui.element("div").classes(f"mac-state {look}"):
        ui.icon(icon)
        text = t(f"models.state.{status.state}")
        if status.ready and key == "flurseg" and screen.c.weights_state:
            text += " · " + t(f"models.weights.{screen.c.weights_state}")
        ui.label(text)
