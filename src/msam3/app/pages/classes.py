"""Classes: FlurSeg's fixed list, and the editor for the SAM 3 prompts and own classes."""

from nicegui import ui

from msam3 import backends, user_classes
from msam3.app import components
from msam3.app.controller import ActionError
from msam3.app.pages.overview import inactive_text

NEW_CLASS_COLOR = "#26B7EF"


def render(screen) -> None:
    t = screen.t
    shown = screen.classes_tab or screen.c.options.backend

    def show(key):
        screen.classes_tab = key
        screen.refresh()

    with ui.element("div").classes("mac-segments").style("align-self:flex-start"):
        for key, model in backends.MODELS.items():
            element = ui.button(on_click=lambda _, key=key: show(key))
            element.props(f'flat no-caps :ripple=false aria-label="{model.title}"')
            element.classes("is-active" if key == shown else "")
            with element:
                ui.label(model.title).classes("mac-segment-name")
                ui.label(t(f"classes.kind.{key}")).classes("mac-segment-sub")
    if shown == "flurseg":
        _fixed_list(screen)
    else:
        _editor(screen)


def _fixed_list(screen) -> None:
    t = screen.t
    components.banner("info", t("classes.fixed_reason"))
    with ui.element("div").classes("mac-card"):
        ui.label(t("classes.fixed_title")).classes("mac-card-title mac-list-head")
        for item in screen.c.run_classes("flurseg").classes:
            with ui.element("div").classes("mac-item"):
                components.swatch(item.color)
                with ui.element("div").classes("mac-item-main"):
                    ui.label(t.class_name(item.name)).classes("mac-item-name")
                    components.note(item.name)
                ui.label(t("result.code", code=item.code)).classes("mac-item-side")
    components.note(t("classes.code_note"))


def _editor(screen) -> None:
    controller, t = screen.c, screen.t
    reason = t(controller.classes_lock) if controller.classes_lock else None
    components.banner("info", t("classes.editable_reason"))
    if reason:
        components.banner("warning", reason)
    if controller.classes_error:
        components.banner("error", f"{t('lock.classes')}\n{controller.classes_error}")
    elif controller.inactive_helpers("sam3"):
        components.banner("warning", inactive_text(screen, "sam3"))
    edit_lock = reason or (t("lock.classes") if controller.classes_error else None)

    with ui.element("div").classes("mac-card"):
        with ui.element("div").classes("mac-row mac-spread mac-list-head"):
            ui.label(t("classes.own_title")).classes("mac-card-title")
            components.button(
                t("action.add_class"), "add", lambda: _class_dialog(screen), "primary", edit_lock
            )
        if not controller.edits.custom:
            with ui.element("div").classes("mac-item"):
                components.note(t("classes.own_empty"))
        resolved = {item.name: item.code for item in controller.sam_classes.classes}
        for item in controller.edits.custom:
            with ui.element("div").classes("mac-item"):
                components.swatch(item.color)
                with ui.element("div").classes("mac-item-main"):
                    ui.label(item.name).classes("mac-item-name")
                    components.note(", ".join(item.prompts))
                code = resolved.get(item.name, item.code)
                ui.label(t("result.code", code=code)).classes("mac-item-side")
                with ui.element("div").classes("mac-item-actions"):
                    components.icon_button(
                        "edit",
                        t("action.edit"),
                        lambda _, item=item: _class_dialog(screen, item),
                        edit_lock,
                    )
                    components.icon_button(
                        "delete_outline",
                        t("action.remove"),
                        lambda _, item=item: screen.act(controller.remove_class, item.name),
                        edit_lock,
                    )

    with ui.element("div").classes("mac-card"):
        with ui.element("div").classes("mac-row mac-spread mac-list-head"):
            ui.label(t("classes.standard_title")).classes("mac-card-title")
            components.button(
                t("action.reset"),
                "restart_alt",
                lambda: screen.confirm(
                    t("classes.reset_title"),
                    t("classes.reset_text"),
                    t("action.reset"),
                    lambda: screen.act(
                        controller.reset_classes, success=t("classes.reset_finished")
                    ),
                ),
                "text",
                reason,
            )
        for item in controller.standard_classes.classes:
            _standard_row(screen, item, edit_lock)
    components.note(t("classes.code_note"))


def _standard_row(screen, item, edit_lock) -> None:
    controller, t = screen.c, screen.t
    enabled = item.name not in controller.edits.disabled
    prompts = controller.edits.prompts.get(item.name, item.prompts)
    with ui.element("div").classes("mac-item"):
        with components.lockable(edit_lock):
            box = ui.checkbox(
                value=enabled,
                on_change=lambda event: screen.act(
                    controller.set_class_enabled, item.name, event.value
                ),
            )
            box.props(f'dense aria-label="{t.class_name(item.name)}"')
            if edit_lock:
                box.disable()
        components.swatch(item.color)
        with ui.element("div").classes("mac-item-main"):
            ui.label(t.class_name(item.name)).classes("mac-item-name")
            text = ", ".join(prompts) if enabled else t("classes.disabled")
            components.note(text)
        ui.label(t("result.code", code=item.code)).classes("mac-item-side")
        components.icon_button(
            "edit",
            t("action.edit_prompts"),
            lambda: _prompts_dialog(screen, item.name, prompts),
            edit_lock,
        )


def _class_dialog(screen, item=None) -> None:
    controller, t = screen.c, screen.t
    with screen.overlay, ui.dialog() as dialog, ui.element("div").classes("mac-card mac-dialog"):
        with ui.element("div").classes("mac-stack"):
            ui.label(t("dialog.edit_class" if item else "dialog.add_class")).classes(
                "mac-card-title"
            )
            name = ui.input(t("dialog.name"), value=item.name if item else "")
            name.props("outlined autofocus").mark("class-name")
            prompts = ui.textarea(
                t("dialog.prompts"), value=", ".join(item.prompts) if item else ""
            ).props("outlined autogrow")
            components.note(t("dialog.prompts_hint"))
            color = ui.color_input(t("dialog.color"), value=item.color if item else NEW_CLASS_COLOR)
            color.props("outlined")
            code = ui.number(
                t("dialog.code"),
                value=item.code if item else None,
                min=user_classes.MIN_CLASS_CODE,
                max=user_classes.MAX_IMPORT_CODE,
                step=1,
            ).props("outlined")
            components.note(t("dialog.code_hint"))

            def save():
                value = code.value
                if value is not None and not float(value).is_integer():
                    screen.notify(t("class_error.code_whole"), "negative")
                    return
                try:
                    controller.save_class(
                        name.value,
                        user_classes.parse_prompts(prompts.value or ""),
                        color.value or "",
                        None if value is None else int(value),
                        item.name if item else None,
                    )
                except ActionError as error:
                    screen.notify(screen.error_text(error), "negative")
                    return
                dialog.close()
                screen.notify(t("classes.saved"))
                screen.refresh()

            with ui.element("div").classes("mac-row"):
                components.button(t("action.save"), "check", save, "primary")
                components.button(t("common.cancel"), "close", dialog.close, "secondary")
    dialog.open()


def _prompts_dialog(screen, name: str, prompts: list[str]) -> None:
    controller, t = screen.c, screen.t
    with screen.overlay, ui.dialog() as dialog, ui.element("div").classes("mac-card mac-dialog"):
        with ui.element("div").classes("mac-stack"):
            ui.label(t.class_name(name)).classes("mac-card-title")
            field = ui.textarea(t("dialog.prompts"), value=", ".join(prompts))
            field.props("outlined autogrow autofocus")
            components.note(t("dialog.prompts_hint"))

            def save():
                try:
                    controller.set_prompts(name, user_classes.parse_prompts(field.value or ""))
                except ActionError as error:
                    screen.notify(screen.error_text(error), "negative")
                    return
                dialog.close()
                screen.notify(t("classes.saved"))
                screen.refresh()

            with ui.element("div").classes("mac-row"):
                components.button(t("action.save"), "check", save, "primary")
                components.button(t("common.cancel"), "close", dialog.close, "secondary")
    dialog.open()
