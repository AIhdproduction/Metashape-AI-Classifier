"""The shell that all pages share, for one connected window."""

import logging
from dataclasses import dataclass

from nicegui import run, ui

from msam3.app import components, theme
from msam3.app.controller import ActionError, Controller
from msam3.app.pages import classes, log, models, overview, settings

LOG = logging.getLogger(__name__)
TECHNOLOGIES = ("Metashape", "FlurSeg", "DINOv3", "EoMT", "SAM 3")


@dataclass(frozen=True)
class Page:
    key: str
    path: str
    icon: str
    render: object
    project_only: bool = False
    wide: bool = False


PAGES = (
    Page("overview", "/", "home", overview.render),
    Page("models", "/models", "memory", models.render),
    Page("classes", "/classes", "category", classes.render),
    Page("settings", "/settings", "tune", settings.render),
    Page("log", "/log", "subject", log.render, project_only=True, wide=True),
)


class Screen:
    def __init__(self, controller: Controller, page: str = "overview"):
        self.c = controller
        self.page = next(item for item in PAGES if item.key == page)
        self.expert_open = False
        self.classes_tab: str | None = None
        self._live = []
        self._revision = -1
        self._ended = False

    @property
    def t(self):
        return self.c.t

    def build(self) -> None:
        ui.add_head_html(theme.style())
        ui.colors(**theme.QUASAR_COLORS)
        ui.dark_mode(True)
        ui.page_title("Metashape AI Classifier")
        with ui.element("div").classes("mac-shell"):
            with ui.element("aside").classes("mac-sidebar"):
                with ui.element("div").classes("mac-brand"):
                    # Our own file, not user content; sanitising would drop the SVG's attributes.
                    ui.html(theme.LOGO_FILE.read_text(encoding="utf-8"), sanitize=False)
                    ui.html(
                        "<b>Metashape</b><span>AI Classifier</span>", sanitize=False, tag="div"
                    ).classes("mac-wordmark")
                self._navigation = ui.element("nav").classes("mac-navlist")
                footer = "<span>|</span>".join(f"<span>{name}</span>" for name in TECHNOLOGIES)
                ui.html(footer, sanitize=False, tag="div").classes("mac-footer")
            with ui.element("main").classes("mac-main"):
                ui.element("div").classes("mac-topbar")
                self._section = ui.element("section").classes("mac-page")
                with self._section:
                    self._title = ui.label().classes("mac-title").props("role=heading aria-level=1")
                    self._content = ui.element("div").classes("mac-cards")
        # Dialogs and toasts live here, so that rebuilding the page does not remove them.
        self.overlay = ui.element("div")
        self.refresh()
        ui.timer(1.0, self.tick)

    # -------------------------------------------------------------------------------- update

    def refresh(self) -> None:
        self._revision = self.c.revision
        if self.page.project_only and self.c.session is None:
            self.page = PAGES[0]
        self._live = []
        self._navigation.clear()
        with self._navigation:
            self._build_navigation()
        if self.page.wide:
            self._section.classes(add="is-wide")
        else:
            self._section.classes(remove="is-wide")
        if self.page.key == "overview" and self.c.session is not None:
            self._title.set_text(self.c.project_name)
        else:
            self._title.set_text(self.t(f"nav.{self.page.key}"))
        self._content.clear()
        with self._content:
            self.page.render(self)
        self.update()

    def _build_navigation(self) -> None:
        for page in PAGES:
            if page.project_only and self.c.session is None:
                continue
            label = self.t(f"nav.{page.key}")
            with ui.element("div").classes("mac-slot") as slot:
                element = ui.button(
                    label, icon=page.icon, on_click=lambda _, key=page.key: self.go(key)
                )
                element.props(f'flat no-caps :ripple=false aria-label="{label}"')
                element.classes("mac-nav" + (" is-active" if page is self.page else ""))
            slot.tooltip(label)

    def live(self, callback) -> None:
        """Register something that changes every second without rebuilding the page."""
        self._live.append(callback)

    def update(self) -> None:
        for callback in self._live:
            callback()

    def tick(self) -> None:
        if self._revision != self.c.revision:
            self.refresh()
        else:
            self.update()
        if not self._ended and (self.c.cancelled_here or self.c.run.final) and not self.c.viewing:
            # The app ends after this; the window must then not show "connection lost".
            self._ended = True
            with self.overlay:
                ui.run_javascript("document.body.classList.add('mac-ended')")
                if self.c.cancelled_here:
                    ui.run_javascript("setTimeout(() => window.close(), 700)")

    def go(self, key: str) -> None:
        self.page = next(page for page in PAGES if page.key == key)
        with self.overlay:
            ui.navigate.history.replace(self.page.path)
            ui.run_javascript("window.scrollTo(0, 0)")
        self.refresh()

    # ------------------------------------------------------------------------------- actions

    def notify(self, text: str, kind: str = "positive") -> None:
        with self.overlay:
            ui.notify(text, type=kind, position="bottom")

    def error_text(self, error: ActionError) -> str:
        return self.t(error.key, **error.values)

    def act(self, action, *args, success: str | None = None, rebuild: bool = True) -> bool:
        try:
            action(*args)
        except ActionError as error:
            self.notify(self.error_text(error), "negative")
            self.refresh()
            return False
        except OSError as error:
            LOG.exception("Action failed")
            self.notify(self.t("error.save", detail=str(error)), "negative")
            return False
        if success:
            self.notify(success)
        if rebuild:
            self.refresh()
        else:
            self.update()
        return True

    async def act_in_background(self, action, success=None) -> None:
        try:
            result = await run.io_bound(action)
        except ActionError as error:
            self.notify(self.error_text(error), "negative")
        else:
            if success:
                self.notify(success(result))
        self.refresh()

    def copy(self, text: str) -> None:
        with self.overlay:
            ui.clipboard.write(text)
        self.notify(self.t("common.copied"))

    def confirm(self, title: str, text: str, label: str, action) -> None:
        with self.overlay, ui.dialog() as dialog, ui.element("div").classes("mac-card mac-dialog"):
            with ui.element("div").classes("mac-stack"):
                ui.label(title).classes("mac-card-title")
                components.note(text)

                def confirmed():
                    dialog.close()
                    action()

                with ui.element("div").classes("mac-row"):
                    components.button(label, "check", confirmed, "primary")
                    components.button(self.t("common.cancel"), "close", dialog.close, "secondary")
        dialog.open()
