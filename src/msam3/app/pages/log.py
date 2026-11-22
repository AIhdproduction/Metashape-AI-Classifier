"""Protokoll: the live log of the session."""

from nicegui import ui

from msam3.app import components

VISIBLE_LINES = 400


def render(screen) -> None:
    controller, t = screen.c, screen.t
    with ui.element("div").classes("mac-row mac-spread"):
        components.note(t("log.hint"))
        components.button(
            t("action.copy_log"), "content_copy", lambda: screen.copy(controller.log), "tool"
        )
    view = ui.log(max_lines=VISIBLE_LINES).classes("mac-log")
    empty = components.note(t("log.empty"))
    shown: list[str] = []

    def update():
        lines = controller.log.splitlines()
        empty.set_visibility(not lines)
        if lines == shown:
            return
        # Logs only grow, so usually only the new lines have to travel to the window.
        if lines[: len(shown)] == shown:
            added = lines[len(shown) :]
        else:
            view.clear()
            added = lines
        for line in added[-VISIBLE_LINES:]:
            view.push(line or " ")
        if not shown and lines:
            # ui.log follows new lines, but opens at the top when it starts filled.
            with screen.overlay:
                ui.run_javascript(
                    f"setTimeout(() => getElement({view.id})?.$refs.qRef"
                    "?.setScrollPosition('vertical', 1e7), 200)"
                )
        shown[:] = lines

    screen.live(update)
    if controller.session is not None:
        components.note(t("log.folder"))
        ui.label(str(controller.session)).classes("mac-code")
