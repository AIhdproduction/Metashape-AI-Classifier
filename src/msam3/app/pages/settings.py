"""Settings of this run. Each model shows only its own options."""

from nicegui import ui

from msam3.app import components, i18n
from msam3.app.pages.overview import inactive_text
from msam3.gui_options import PROFILE_NAMES
from msam3.segmentation.windows import LONG_SIDE


def render(screen) -> None:
    controller, t = screen.c, screen.t
    if controller.session is None:
        components.banner("info", t("settings.standalone"))
        _language(screen)
        return
    reason = t(controller.options_lock) if controller.options_lock else None
    if reason:
        components.banner("warning", reason)
    _profile(screen, reason)
    if controller.options.backend == "flurseg":
        _flurseg(screen, reason)
    else:
        _sam(screen, reason)
    _assists(screen, reason)
    _filters(screen, reason)
    _expert(screen, reason)
    _language(screen)


def _card(title: str, hint: str):
    card = ui.element("div").classes("mac-card mac-stack")
    with card:
        ui.label(title).classes("mac-card-title")
        components.note(hint)
    return card


def _changed(screen, name: str):
    return lambda event: screen.act(screen.c.set_option, name, event.value, rebuild=False)


def _select(screen, label: str, name: str, values, reason, text=str) -> None:
    current = getattr(screen.c.options, name)
    choices = dict.fromkeys([*values, current])
    try:
        choices = sorted(choices)
    except TypeError:  # Texts and None have no common order.
        choices = list(choices)
    with components.lockable(reason):
        field = ui.select(
            {value: text(value) for value in choices},
            value=current,
            label=label,
            on_change=_changed(screen, name),
        ).props("outlined")
        if reason:
            field.disable()


def _number(screen, label: str, name: str, reason, low, high, step) -> None:
    with components.lockable(reason):
        field = ui.number(
            label,
            value=getattr(screen.c.options, name),
            min=low,
            max=high,
            step=step,
            on_change=_changed(screen, name),
        ).props("outlined")
        if reason:
            field.disable()


def _switch(screen, label: str, name: str, reason) -> None:
    with components.lockable(reason):
        field = ui.checkbox(
            label, value=getattr(screen.c.options, name), on_change=_changed(screen, name)
        )
        if reason:
            field.disable()


def _profile(screen, reason) -> None:
    controller, t = screen.c, screen.t
    backend = controller.options.backend
    with _card(t("settings.profile"), t("settings.profile_hint")):
        shown = controller.options.profile

        def choose(event):
            if event.value and event.value != controller.options.profile:
                screen.act(controller.select_profile, event.value)

        with components.lockable(reason):
            toggle = ui.toggle(
                {name: t(f"profile.{name}") for name in PROFILE_NAMES},
                value=shown if shown in PROFILE_NAMES else None,
                on_change=choose,
            )
            toggle.props("no-caps unelevated :ripple=false").classes("mac-toggle")
            if reason:
                toggle.disable()
        explanation = components.note("")

        def update():
            profile = controller.options.profile
            if profile == "custom":
                toggle.set_value(None)
                explanation.set_text(t("profile.custom_hint"))
            else:
                toggle.set_value(profile)
                explanation.set_text(t(f"profile.{backend}_{profile}"))

        screen.live(update)


def _flurseg(screen, reason) -> None:
    t = screen.t

    def resolution(name):
        limit = LONG_SIDE[name]
        return t("settings.original") if limit is None else t("settings.long_side", pixels=limit)

    with _card(t("settings.flurseg"), t("settings.flurseg_hint")):
        with ui.element("div").classes("mac-fields"):
            _select(
                screen, t("settings.window_batch"), "flurseg_batch_size", (1, 2, 4, 8, 16), reason
            )
            _select(screen, t("settings.min_views"), "flurseg_min_views", (1, 2, 3), reason)
        components.note(
            t("settings.image_size", value=resolution(screen.c.options.flurseg_resolution))
        )


def _sam(screen, reason) -> None:
    controller, t = screen.c, screen.t

    def percent(value):
        return f"{round(value * 100)} %"

    with _card(t("settings.sam"), t("settings.sam_hint")):
        with ui.element("div").classes("mac-fields"):
            _select(
                screen,
                t("settings.image_scale"),
                "sam_image_scale",
                (0.5, 0.75, 1.0),
                reason,
                percent,
            )
            _select(
                screen,
                t("settings.overlap"),
                "sam_tile_overlap",
                (0.1, 0.2, 0.3),
                reason,
                percent,
            )
            _select(screen, t("settings.min_views"), "sam_min_views", (1, 2, 3), reason)
            _select(
                screen,
                t("settings.prompt_batch"),
                "sam_prompt_batch_size",
                (4, 8, 12, 16),
                reason,
            )

        async def auto():
            def success(result):
                batch, free, total = result
                values = {"free": f"{free:.1f}", "total": f"{total:.1f}"}
                return t("settings.auto_done", batch=batch, **values)

            await screen.act_in_background(controller.auto_batch, success)

        components.button(t("action.auto_batch"), "speed", auto, "text", reason)


def _assists(screen, reason) -> None:
    t = screen.t
    with _card(t("settings.assists"), t("settings.assists_hint")):
        _switch(screen, t("settings.ground_assist"), "ground_assist", reason)
        _switch(screen, t("settings.vegetation_assist"), "vegetation_assist", reason)


def _filters(screen, reason) -> None:
    controller, t = screen.c, screen.t
    with _card(t("settings.filters"), t("settings.filters_hint")):
        _switch(screen, t("settings.postprocess"), "postprocess", reason)
        _switch(screen, t("settings.vegetation"), "vegetation", reason)
        _switch(screen, t("settings.road"), "road", reason)
        _switch(screen, t("settings.smoothing"), "smoothing", reason)
        if controller.options.backend == "sam3":
            warning = components.banner("warning", inactive_text(screen))
            screen.live(lambda: warning.set_visibility(bool(controller.inactive_helpers())))


def _expert(screen, reason) -> None:
    t = screen.t
    backend = screen.c.options.backend
    with ui.expansion(t("settings.expert"), value=screen.expert_open) as expansion:
        with ui.element("div").classes("mac-stack"):
            components.note(t("settings.expert_hint"))
            with ui.element("div").classes("mac-fields"):
                if backend == "flurseg":
                    _select(
                        screen,
                        t("settings.precision"),
                        "flurseg_precision",
                        ("fp16", "fp32"),
                        reason,
                    )
                    background = "flurseg_background_threshold"
                    confidence = "flurseg_min_mask_confidence"
                else:
                    _number(
                        screen,
                        t("settings.instance_threshold"),
                        "sam_instance_threshold",
                        reason,
                        0,
                        1,
                        0.05,
                    )
                    background = "sam_background_threshold"
                    confidence = "sam_min_mask_confidence"
                _number(screen, t("settings.background"), background, reason, 0, 1, 0.05)
                _number(screen, t("settings.mask_confidence"), confidence, reason, 0, 1, 0.05)
                _number(screen, t("settings.min_share"), "min_share", reason, 0, 1, 0.05)
                _number(screen, t("settings.voxel"), "voxel_size_m", reason, 0.005, 1, 0.005)
                _number(screen, t("settings.interior"), "mask_interior_px", reason, 0, 8, 1)
    expansion.classes("mac-expansion").props(":duration=0")
    expansion.on_value_change(lambda event: setattr(screen, "expert_open", event.value))


def _language(screen) -> None:
    controller, t = screen.c, screen.t
    with _card(t("settings.language"), t("settings.language_hint")):
        with ui.element("div").classes("mac-row"):
            toggle = ui.toggle(
                i18n.LANGUAGES,
                value=t.language,
                on_change=lambda event: screen.act(controller.set_language, event.value),
            )
            toggle.props("no-caps unelevated :ripple=false").classes("mac-toggle")
