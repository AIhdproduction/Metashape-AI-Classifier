"""Small building blocks that every page shares."""

from contextlib import contextmanager
from html import escape

from nicegui import ui

BANNER_ICONS = {"info": "info", "warning": "warning", "error": "error"}


def note(text: str, classes: str = "mac-note"):
    return ui.label(text).classes(classes)


def banner(kind: str, text: str):
    element = ui.element("div").classes(f"mac-banner is-{kind}")
    with element:
        ui.icon(BANNER_ICONS[kind])
        ui.label(text).classes("mac-banner-text")
    return element


@contextmanager
def lockable(reason: str | None):
    """A disabled Quasar control swallows hover, so its tooltip lives on the surrounding slot."""
    slot = ui.element("div").classes("mac-slot")
    with slot:
        yield slot
    if reason:
        slot.classes("mac-locked")
        slot.tooltip(reason)


def button(label: str, icon: str | None, action, kind: str = "text", reason: str | None = None):
    looks = {
        "primary": ("unelevated color=primary", "mac-primary"),
        "secondary": ("outline color=primary", "mac-secondary"),
        "text": ("flat color=primary", "mac-text-button"),
        "tool": ("unelevated", "mac-tool"),
    }
    props, classes = looks[kind]
    color = None if kind == "tool" else "primary"
    with lockable(reason):
        element = ui.button(label, icon=icon, on_click=action, color=color)
        element.props(f"{props} no-caps :ripple=false").classes(classes)
        if reason:
            element.disable()
    return element


def icon_button(icon: str, label: str, action, reason: str | None = None):
    with lockable(reason):
        element = ui.button(icon=icon, on_click=action)
        element.props(f'flat dense color=primary :ripple=false aria-label="{escape(label)}"')
        if reason:
            element.disable()
        else:
            element.tooltip(label)
    return element


def swatch(color: str):
    return ui.element("div").classes("mac-swatch").style(f"background:{color}")


def link(text: str, url: str):
    with ui.link(target=url, new_tab=True).classes("mac-link") as element:
        ui.icon("open_in_new")
        ui.label(text)
    return element
