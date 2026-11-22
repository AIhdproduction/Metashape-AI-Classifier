"""Overview: workspace on the left, the four real steps of the workflow on the right."""

import os
from datetime import datetime
from html import escape
from pathlib import Path

from nicegui import ui

from msam3 import backends
from msam3.app import components
from msam3.gui_options import FLURSEG_SECONDS_PER_WINDOW, format_duration
from msam3.gui_progress import duration


def render(screen) -> None:
    controller, t = screen.c, screen.t
    if controller.viewing:
        with ui.element("div").classes("mac-row mac-spread"):
            components.note(t("overview.viewing"))
            components.button(
                t("action.close_project"),
                "arrow_back",
                lambda: screen.act(controller.close_recent),
            )
    with ui.element("div").classes("mac-home"):
        with ui.element("div").classes("mac-cards"):
            if controller.session is None:
                _empty(screen)
                _recent(screen)
            elif controller.run.state == "ready":
                _ready(screen)
            elif controller.run.state == "running":
                _running(screen)
            elif controller.run.state == "done":
                _result(screen)
            else:
                _stopped(screen)
        _workflow(screen)


def model_switch(screen, reason: str | None) -> None:
    controller, t = screen.c, screen.t
    with components.lockable(reason):
        with ui.element("div").classes("mac-segments"):
            for key, model in backends.MODELS.items():
                element = ui.button(
                    on_click=lambda _, key=key: screen.act(controller.select_model, key)
                )
                element.props(f'flat no-caps :ripple=false aria-label="{model.title}"')
                element.classes("is-active" if key == controller.options.backend else "")
                with element:
                    ui.label(model.title).classes("mac-segment-name")
                    ui.label(t(f"model.{key}.short")).classes("mac-segment-sub")
                if reason:
                    element.disable()


def summary(screen) -> str:
    controller, t = screen.c, screen.t
    return t(
        "overview.summary",
        profile=t(f"profile.{controller.options.profile}"),
        classes=controller.shown_class_count(),
        label=controller.imported_label(),
    )


def _empty(screen) -> None:
    t = screen.t
    with ui.element("div").classes("mac-zone"):
        ui.icon("grain")
        ui.label(t("overview.empty_title")).classes("mac-zone-title")
        components.note(t("overview.empty_hint"), "mac-text")
        components.banner("info", t("lock.standalone"))
        components.button(
            t("action.start"), "play_arrow", lambda: None, "primary", t("lock.standalone")
        )


def _recent(screen) -> None:
    controller, t = screen.c, screen.t
    with ui.element("div").classes("mac-card mac-list"):
        ui.label(t("overview.recent")).classes("mac-card-title mac-list-head")
        if not controller.recent:
            with ui.element("div").classes("mac-item"):
                components.note(t("overview.recent_empty"))
        for entry in controller.recent:
            with ui.element("div").classes("mac-item"):
                with ui.element("div").classes("mac-item-main"):
                    path = str(entry["project"])
                    ui.label(Path(path.replace("\\", "/")).stem).classes("mac-item-name")
                    ui.label(path).classes("mac-item-path").props(f'title="{escape(path)}"')
                    when = datetime.fromtimestamp(entry["at"]).strftime("%d.%m.%Y %H:%M")
                    state = t(f"recent.{entry['phase'] or 'open'}")
                    components.note(f"{when} · {entry['model']} · {state}")
                components.button(
                    t("action.open"),
                    "folder_open",
                    lambda _, session=entry["session"]: screen.act(controller.open_recent, session),
                )


def _ready(screen) -> None:
    controller, t = screen.c, screen.t
    backend = controller.options.backend
    start_lock = controller.start_lock
    if start_lock:
        with ui.element("div").classes("mac-stack"):
            components.banner("warning", t(start_lock))
            if start_lock.startswith("lock.model_"):
                components.button(
                    t("action.to_models"), "download", lambda: screen.go("models"), "text"
                )
    if backend == "sam3" and controller.inactive_helpers():
        components.banner("warning", inactive_text(screen))
    with ui.element("div").classes("mac-zone"):
        ui.icon("grain")
        ui.label(t("overview.ready_title")).classes("mac-zone-title")
        components.note(t("overview.photos", count=t.number(controller.image_count)))
        model_switch(screen, None)
        components.note(t(f"model.{backend}.choice"), "mac-text")
        components.note(t("model.speed_note"))
        summary_label = components.note(summary(screen), "mac-text")
        estimate = ui.label().classes("mac-estimate")
        source = components.note("")

        def update():
            summary_label.set_text(summary(screen))
            value = controller.estimate()
            if value is None:
                estimate.set_text(t("estimate.unknown"))
                source.set_text("")
                return
            estimate.set_text(
                t("estimate.value", time=format_duration(value.seconds, t("common.about")))
            )
            kind = "history" if value.from_history else "default"
            unit = "windows" if backend == "flurseg" else "tiles"
            source.set_text(
                t(
                    f"estimate.{unit}_{kind}",
                    count=t.number(value.units),
                    rate=FLURSEG_SECONDS_PER_WINDOW,
                )
            )

        screen.live(update)
        with ui.element("div").classes("mac-row"):
            components.button(
                t("action.start"),
                "play_arrow",
                lambda: screen.act(controller.start),
                "primary",
                t(start_lock) if start_lock else None,
            )
            components.button(
                t("action.cancel"),
                "close",
                lambda: screen.act(controller.cancel_before_start),
                "secondary",
            )


def inactive_text(screen, backend: str | None = None) -> str:
    t = screen.t
    lines = [t("inactive.title")]
    for key, names in screen.c.inactive_helpers(backend).items():
        missing = ", ".join(t.class_name(name) for name in names)
        lines.append(t("inactive.line", helper=t(f"inactive.{key}"), classes=missing))
    return "\n".join(lines)


def _running(screen) -> None:
    controller, t = screen.c, screen.t
    run = controller.run
    components.banner("info", t("running.banner"))
    with ui.element("div").classes("mac-zone"):
        ui.icon("blur_on")
        ui.label(t(f"running.{run.phase}")).classes("mac-zone-title")
        detail = components.note("", "mac-text mac-number")
        remaining = components.note("", "mac-note mac-number")
        model_switch(screen, t("lock.running"))
        components.note(summary(screen))

        def update():
            view = controller.run
            images = view.images
            detail.set_text(t("running.keep_open"))
            remaining.set_text("")
            if view.phase != "classify":
                return
            if images.finishing:
                detail.set_text(t("running.finishing"))
            elif images.total and not images.name:
                # Planned, but the first image is still in the model.
                detail.set_text(t("running.first", total=images.total))
            elif images.total:
                detail.set_text(
                    t("running.images", done=images.done, total=images.total, name=images.name)
                )
                if images.remaining_seconds is not None:
                    remaining.set_text(
                        t("running.remaining", time=duration(images.remaining_seconds))
                    )
            else:
                detail.set_text(t("running.loading"))

        screen.live(update)
        if run.cancel_requested:
            reason = t("cancel.requested")
        elif run.can_cancel:
            reason = None
        else:
            reason = t("cancel.locked")

        async def cancel():
            await screen.act_in_background(controller.cancel_run)

        label = t("action.cancelling") if run.cancel_requested else t("action.cancel")
        components.button(label, "stop", cancel, "secondary", reason)


def _result(screen) -> None:
    controller, t = screen.c, screen.t
    result = controller.run.result
    label = result.get("label") or controller.imported_label()
    components.banner("info", t("result.banner", label=label))
    with ui.element("div").classes("mac-card"):
        with ui.element("div").classes("mac-row mac-spread mac-list-head"):
            ui.label(t("result.title")).classes("mac-card-title")
            title = result.get("model", {}).get("title", "")
            total = t.number(result.get("point_count", 0))
            components.note(t("result.total", model=title, count=total), "mac-note mac-number")
        with ui.element("div").classes("mac-list"):
            points = max(1, result.get("point_count", 0))
            for item in result.get("classes", []):
                if not item["count"]:
                    continue
                with ui.element("div").classes("mac-item"):
                    components.swatch(item["color"])
                    with ui.element("div").classes("mac-item-main"):
                        ui.label(t.class_name(item["name"])).classes("mac-item-name")
                        components.note(t("result.code", code=item["code"]))
                    share = item["count"] / points * 100
                    share_text = f"{share:.1f} %" if share >= 0.05 else "< 0.1 %"
                    with ui.element("div").classes("mac-item-side"):
                        ui.label(t.number(item["count"]))
                        ui.label(share_text)
        outputs = result.get("outputs", [])
        with ui.element("div").classes("mac-stack mac-actions"):
            if outputs:
                components.note(t("result.output"))
                for output in outputs:
                    ui.label(output).classes("mac-code")
            with ui.element("div").classes("mac-row"):
                if outputs:
                    components.button(
                        t("action.open_folder"),
                        "folder_open",
                        lambda: _open_folder(screen, Path(outputs[0]).parent),
                    )
                    components.button(
                        t("action.copy_path"),
                        "content_copy",
                        lambda: screen.copy("\n".join(outputs)),
                    )
                components.button(t("action.to_log"), "subject", lambda: screen.go("log"))


def _open_folder(screen, folder: Path) -> None:
    if not hasattr(os, "startfile") or not folder.is_dir():
        screen.notify(screen.t("error.folder"), "negative")
        return
    os.startfile(folder)


def _stopped(screen) -> None:
    controller, t = screen.c, screen.t
    run = controller.run
    cancelled = run.state == "cancelled"
    if cancelled:
        components.banner("warning", t("cancelled.banner"))
    else:
        message = t("failed.interrupted") if run.interrupted else run.message
        components.banner("error", message or t("failed.unknown"))
    if run.worker_running:
        components.banner("warning", t("worker.running"))
    with ui.element("div").classes("mac-zone"):
        ui.icon("block" if cancelled else "error_outline")
        ui.label(t("cancelled.title" if cancelled else "failed.title")).classes("mac-zone-title")
        components.note(t("cancelled.hint" if cancelled else "failed.hint"), "mac-text")
        if run.worker_running:

            async def stop():
                await screen.act_in_background(controller.stop_worker)

            components.button(t("action.stop_worker"), "stop", stop, "secondary")
        components.button(t("action.to_log"), "subject", lambda: screen.go("log"))


def _workflow(screen) -> None:
    card = ui.html("", sanitize=False).classes("mac-card mac-flow")
    shown = [None]

    def update():
        html = workflow_html(screen)
        if html != shown[0]:
            shown[0] = html
            card.set_content(html)

    screen.live(update)


def workflow_html(screen) -> str:
    """Generated from fixed texts and numbers; everything is escaped."""
    controller, t = screen.c, screen.t
    run = controller.run
    parts = [f'<div class="mac-flow-title">{escape(t("flow.title"))}</div>']
    parts.append('<div class="mac-steps">')
    for number, step in enumerate(run.steps, start=1):
        status = t(f"flow.{step.state}")
        bar = ""
        if step.state == "active" and step.percent is not None:
            status = f"{status} · {step.percent} %"
            bar = f'<div class="mac-bar"><i style="width:{step.percent}%"></i></div>'
        elif step.state == "done":
            status = f"{status} · 100 %"
        parts.append(
            f'<div class="mac-step is-{step.state}"><span class="mac-step-no">{number}</span>'
            f'<span class="mac-step-name">{escape(t(f"step.{step.key}"))}</span>'
            f'<span class="mac-step-status">{escape(status)}</span>{bar}</div>'
        )

    def row(name: str, value: str) -> str:
        cells = f"<span>{escape(name)}</span><strong>{escape(value)}</strong>"
        return f'<div class="mac-total">{cells}</div>'

    def seconds(value) -> str:
        return duration(value) if value is not None else "–"

    parts.append('</div><div class="mac-totals">')
    parts.append(row(t("flow.total"), seconds(run.total_seconds)))
    parts.append(f'<div class="mac-total-head">{escape(t("flow.per_step"))}</div>')
    parts += [row(t(f"step.{step.key}"), seconds(step.seconds)) for step in run.steps]
    parts.append(f'<div class="mac-total-head">{escape(t("flow.storage"))}</div>')
    for key in ("work", "free"):
        parts.append(row(t(f"flow.storage_{key}"), size_text(controller.storage.get(key))))
    parts.append("</div>")
    return "".join(parts)


def size_text(value: int | None) -> str:
    if value is None:
        return "–"
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GB"
    return f"{value / 1024**2:.1f} MB"
