"""The SAMcloud design system: every colour and size of the app is defined here."""

from pathlib import Path

LOGO_FILE = Path(__file__).resolve().parent / "static/logo.svg"

COLORS = {
    "bg": "#09131d",
    "panel": "#0e1c28",
    "dropzone": "#0c1924",
    "canvas": "#050b12",
    "line": "#20394b",
    "line-soft": "rgba(113,154,181,.18)",
    "line-viewer": "#28475b",
    "line-field": "#456176",
    "line-strong": "#47718a",
    "text": "#eaf2f8",
    "text-2": "#c9d8e4",
    "muted": "#96aabe",
    "faint": "#7890a5",
    "icon": "#a8bfd1",
    "accent": "#26b7ef",
    "accent-light": "#8cdbfa",
    "on-accent": "#06202f",
    "success": "#4bc692",
    "error": "#ef6b73",
    "overlay": "rgba(9,19,29,.86)",
    "white": "#ffffff",
    "sidebar-top": "#0a1824",
    "sidebar-bottom": "#0a1520",
    "nav-hover": "rgba(55,137,182,.14)",
    "nav-active": "rgba(36,150,207,.22)",
    "glow": "rgba(24,93,126,.14)",
    "zone-icon": "#c2d7e6",
    "step-ring": "#5f7a90",
    "step-link": "#537189",
    "step-active-line": "#24719b",
    "step-active": "rgba(24,105,150,.14)",
    "toolbar": "#102131",
    "info-bg": "rgba(38,183,239,.10)",
    "info-line": "#24719b",
    "info-text": "#bfe9fb",
    "warning-bg": "rgba(255,183,77,.13)",
    "warning-line": "#a8752b",
    "warning-text": "#ffd59a",
    "error-bg": "rgba(239,107,115,.13)",
    "error-line": "#9b3c43",
    "error-text": "#ffc2c6",
}

QUASAR_COLORS = {
    "primary": COLORS["accent"],
    "positive": COLORS["success"],
    "negative": COLORS["error"],
    "dark": COLORS["panel"],
    "dark_page": COLORS["bg"],
}

_RULES = """
html, body { background:var(--bg) !important; color:var(--text); }
body { font-family:Roboto, sans-serif; font-size:14px; }
*, *::before, *::after { box-sizing:border-box; }
.nicegui-content { width:100%; padding:0; gap:0; }
.mac-shell { display:flex; width:100%; min-height:100vh; background:var(--bg); color:var(--text); }
.mac-shell .q-icon, .q-menu .q-icon { user-select:none; }

/* Sidebar */
.mac-sidebar { width:260px; flex:none; height:100vh; position:sticky; top:0; overflow-y:auto;
  display:flex; flex-direction:column; padding:30px 16px 22px; border-right:1px solid var(--line);
  background:linear-gradient(180deg, var(--sidebar-top) 0%, var(--sidebar-bottom) 100%); }
.mac-brand { display:flex; align-items:center; gap:10px; margin:0 6px 38px; }
.mac-brand svg { width:34px; height:34px; flex:none; }
.mac-wordmark { display:flex; flex-direction:column; font-size:32px; line-height:1;
  letter-spacing:-1.4px; white-space:nowrap; color:var(--white); }
.mac-wordmark b { color:var(--accent); font-weight:800; }
.mac-navlist { display:flex; flex-direction:column; gap:3px; width:100%; }
.mac-slot { display:flex; max-width:100%; }
.mac-navlist .mac-slot { width:100%; }
.mac-nav { width:100%; min-height:52px; padding:0 14px; border-radius:8px; font-size:15px;
  font-weight:500; }
.mac-nav .q-btn__content { justify-content:center; flex-wrap:nowrap; }
.mac-nav .q-icon { margin-right:13px; font-size:21px; color:var(--icon); }
.mac-nav:hover { background:var(--nav-hover); }
.mac-nav.is-active { background:var(--nav-active); border-radius:0 8px 8px 0; }
/* q-btn::before carries the outline, so the active edge is drawn by its own layer. */
.mac-nav.is-active { box-shadow:inset 3px 0 0 var(--accent); }
.mac-nav.is-active .q-icon { color:var(--accent); }
.mac-nav.disabled { opacity:.42 !important; cursor:not-allowed !important; }
.mac-footer { margin:auto 11px 0; padding-top:24px; display:flex; flex-wrap:wrap; gap:4px 6px;
  color:var(--faint); font-size:10px; letter-spacing:1.5px; line-height:1.5; }
.mac-footer span { white-space:nowrap; }

/* Page */
.mac-main { flex:1 1 auto; min-width:0; }
.mac-topbar { height:32px; border-bottom:1px solid var(--line-soft); }
.mac-page { width:100%; max-width:1220px; margin:0 auto; padding:46px clamp(22px,4vw,58px) 64px; }
.mac-page.is-wide { max-width:none; padding:18px clamp(12px,1.6vw,26px) 24px; }
.mac-title { margin:0 0 31px; color:var(--text); font-size:clamp(30px,3vw,42px); font-weight:650;
  letter-spacing:-.8px; line-height:1.12; overflow-wrap:anywhere; }
.mac-page.is-wide .mac-title { font-size:clamp(24px,2.2vw,32px); margin-bottom:18px; }
.mac-section { color:var(--text); font-size:20px; font-weight:600; letter-spacing:-.2px; }
.mac-card-title { color:var(--text); font-size:18px; font-weight:600; line-height:1.3;
  overflow-wrap:anywhere; }
.mac-text { color:var(--text-2); font-size:14px; line-height:1.5; overflow-wrap:anywhere; }
.mac-note { color:var(--muted); font-size:12.5px; line-height:1.5; overflow-wrap:anywhere; }
.mac-number { font-variant-numeric:tabular-nums; }
.mac-stack { display:flex; flex-direction:column; align-items:stretch; gap:12px; width:100%;
  min-width:0; }
.mac-cards { display:flex; flex-direction:column; gap:22px; width:100%; min-width:0; }
.mac-row { display:flex; flex-wrap:wrap; align-items:center; gap:10px; max-width:100%; }
.mac-spread { justify-content:space-between; width:100%; }
.mac-card { width:100%; min-width:0; padding:22px; background:var(--panel);
  border:1px solid var(--line); border-radius:9px; box-shadow:none; }
.mac-home { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:22px; width:100%;
  align-items:start; }
.mac-two { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:22px; width:100%;
  align-items:stretch; }
.mac-fields { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(240px,100%),1fr));
  gap:16px; width:100%; }
.mac-fields > * { min-width:0; width:100%; }
.mac-fields > .mac-slot > * { flex:1; min-width:0; }

/* Workspace */
.mac-zone { width:100%; min-width:0; min-height:390px; padding:42px 30px; display:flex;
  flex-direction:column; align-items:center; justify-content:center; gap:14px; text-align:center;
  border:1px dashed var(--line-strong); border-radius:9px;
  background:radial-gradient(circle at 50% 25%, var(--glow), transparent 45%), var(--dropzone); }
.mac-zone > .q-icon { color:var(--zone-icon); font-size:64px; }
.mac-zone-title { color:var(--text); font-size:21px; font-weight:500; overflow-wrap:anywhere; }
.mac-zone .mac-note, .mac-zone .mac-text { max-width:560px; }
.mac-zone .mac-row { justify-content:center; }
.mac-estimate { color:var(--text); font-size:15px; font-weight:500;
  font-variant-numeric:tabular-nums; }

/* Buttons */
.q-btn { text-transform:none; letter-spacing:0; }
.q-btn::before { box-shadow:none !important; }
.mac-primary, .mac-secondary { min-height:46px; padding:0 18px; border-radius:7px;
  font-size:14px; }
.mac-primary { font-weight:700; color:var(--white) !important; }
.mac-text-button { color:var(--accent); padding:4px 8px; min-height:34px; }
.mac-text-button .q-icon { font-size:18px; }
.mac-tool { background:var(--toolbar); color:var(--text) !important; padding:6px 10px;
  min-height:0; border:1px solid var(--line-strong); border-radius:5px; }
.mac-primary .q-icon, .mac-secondary .q-icon { font-size:20px; }
.mac-tool .q-icon { font-size:17px; }

/* Segmented switch */
.mac-segments { display:flex; max-width:100%; border:1px solid var(--line-strong);
  border-radius:7px; overflow:hidden; }
.mac-segments .q-btn { border-radius:0; min-height:46px; padding:4px 18px; color:var(--text-2)
  !important; }
.mac-segments .q-btn + .q-btn { border-left:1px solid var(--line-strong); }
.mac-segments .q-btn .q-btn__content { flex-direction:column; line-height:1.25; }
.mac-segments .q-btn.is-active { background:var(--accent); color:var(--on-accent) !important; }
.mac-segment-name { font-size:15px; font-weight:700; }
.mac-segment-sub { font-size:12px; }
.mac-toggle { border:1px solid var(--line-strong); border-radius:7px; flex-wrap:wrap;
  max-width:100%; overflow:hidden; }
.mac-toggle .q-btn { min-height:40px; padding:4px 16px; }
.mac-toggle .q-btn[aria-pressed="true"] { color:var(--on-accent) !important; }

/* Banners */
.mac-banner { display:flex; align-items:flex-start; gap:10px; width:100%; padding:10px 14px;
  border:1px solid; border-radius:7px; font-size:14px; line-height:1.45; text-align:left; }
.mac-banner .q-icon { font-size:19px; margin-top:1px; flex:none; }
.mac-banner-text { min-width:0; overflow-wrap:anywhere; white-space:pre-line; }
.mac-banner.is-info { background:var(--info-bg); border-color:var(--info-line);
  color:var(--info-text); }
.mac-banner.is-warning { background:var(--warning-bg); border-color:var(--warning-line);
  color:var(--warning-text); }
.mac-banner.is-error { background:var(--error-bg); border-color:var(--error-line);
  color:var(--error-text); }

/* Workflow */
.mac-flow { padding:25px 22px; }
.mac-flow-title { margin-bottom:16px; color:var(--text); font-size:19px; font-weight:600; }
.mac-step { position:relative; min-height:66px; padding:10px 0 10px 50px; color:var(--text-2); }
.mac-step:not(:last-child)::before { content:""; position:absolute; left:17px; top:43px;
  bottom:-7px; border-left:1px dashed var(--step-link); }
.mac-step-no { position:absolute; left:0; top:7px; width:35px; height:35px; display:grid;
  place-items:center; border:1px solid var(--step-ring); border-radius:50%; color:var(--text-2);
  font-size:13px; font-variant-numeric:tabular-nums; }
.mac-step-name { display:block; font-size:15px; }
.mac-step-status { display:block; margin-top:3px; color:var(--muted); font-size:12px;
  font-variant-numeric:tabular-nums; }
.mac-step.is-active { margin:0 -9px; padding-left:59px; padding-right:9px;
  border:1px solid var(--step-active-line); border-radius:7px; background:var(--step-active); }
.mac-step.is-active .mac-step-no { left:8px; border-color:var(--accent);
  background:var(--accent); color:var(--on-accent); }
.mac-step.is-active:not(:last-child)::before { left:25px; }
.mac-step.is-done .mac-step-no { border-color:var(--success); color:var(--success); }
.mac-step.is-error .mac-step-no { border-color:var(--error); color:var(--error); }
.mac-bar { height:3px; margin-top:7px; border-radius:2px; overflow:hidden;
  background:var(--line); }
.mac-bar i { display:block; height:100%; background:var(--accent); }
.mac-totals { margin-top:16px; padding-top:14px; border-top:1px solid var(--line); }
.mac-total { display:flex; justify-content:space-between; gap:10px; padding:3px 0; font-size:12px;
  color:var(--muted); }
.mac-total strong { color:var(--text); font-weight:500; font-variant-numeric:tabular-nums;
  text-align:right; white-space:nowrap; }
.mac-total-head { margin-top:10px; color:var(--text-2); font-size:12px; }

/* Lists */
.mac-list { width:100%; }
.mac-list-head { padding-bottom:14px; border-bottom:1px solid var(--line); }
.mac-item { display:flex; align-items:center; gap:12px; width:100%; min-width:0; padding:12px 2px;
  border-bottom:1px solid var(--line-soft); text-align:left; }
.mac-item:last-child { border-bottom:0; }
.mac-item-main { flex:1 1 auto; min-width:0; }
.mac-item-name { color:var(--text); font-size:14px; font-weight:600; overflow-wrap:anywhere; }
.mac-item-path { color:var(--muted); font-size:13px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.mac-item-side { flex:none; color:var(--muted); font-size:12.5px; text-align:right;
  font-variant-numeric:tabular-nums; }
.mac-swatch { width:14px; height:14px; flex:none; border:1px solid var(--line-soft);
  border-radius:2px; }
.mac-item-actions { display:flex; flex:none; }
.mac-actions { margin-top:14px; padding-top:12px; border-top:1px solid var(--line); }

/* Models */
.mac-model { display:flex; flex-direction:column; gap:14px; }
.mac-model.is-selected { border-color:var(--accent); }
.mac-model-select { margin-top:auto; padding-top:6px; }
.mac-state { display:flex; align-items:center; gap:6px; font-size:12.5px; color:var(--muted); }
.mac-state .q-icon { font-size:18px; }
.mac-state.is-ready { color:var(--success); }
.mac-state.is-problem { color:var(--error); }
.mac-fact { display:flex; justify-content:space-between; gap:14px; width:100%; padding:6px 0;
  border-bottom:1px solid var(--line-soft); font-size:13px; color:var(--muted); }
.mac-fact > :last-child { color:var(--text); text-align:right; }
.mac-code { width:100%; padding:8px; background:var(--canvas); border:1px solid var(--line);
  border-radius:6px; color:var(--text-2); font:11px/1.6 ui-monospace, Consolas, monospace;
  white-space:pre-wrap; overflow-wrap:anywhere; user-select:text; }
.mac-output { border-color:var(--success); gap:10px; }
.mac-output > .mac-row > .q-icon { color:var(--success); font-size:22px; }
.mac-output-path { width:100%; padding:10px 12px; background:var(--canvas);
  border:1px solid var(--line-strong); border-radius:6px; color:var(--text);
  font:600 14px/1.5 ui-monospace, Consolas, monospace; overflow-wrap:anywhere; user-select:all; }
.mac-link { color:var(--accent); font-size:14px; text-decoration:none; display:inline-flex;
  align-items:center; gap:6px; }
.mac-link .q-icon { font-size:18px; }

/* Log */
.mac-log.nicegui-log { width:100%; height:220px; max-height:220px; padding:8px;
  background:var(--canvas); border:1px solid var(--line); border-radius:6px; color:var(--text-2);
  font:11px/1.55 ui-monospace, Consolas, monospace; }
.mac-log .q-scrollarea__content { width:100%; gap:0; padding:0; align-items:stretch; }
.mac-log .q-scrollarea__content > * { white-space:pre-wrap; overflow-wrap:anywhere; }

/* Quasar fields, menus, dialogs */
.q-field__native, .q-field__input, .q-menu .q-item { color:var(--text) !important; }
.q-field__label { color:var(--muted) !important; }
.q-select__dropdown-icon { color:var(--icon) !important; }
.q-field--outlined .q-field__control::before { border-color:var(--line-field) !important; }
.q-field--outlined .q-field__control:hover::before { border-color:var(--accent) !important; }
.q-menu { background:var(--panel) !important; border:1px solid var(--line);
  box-shadow:none !important; }
.q-checkbox__label { color:var(--text); font-size:14px; }
.mac-expansion { width:100%; background:var(--panel); border:1px solid var(--line);
  border-radius:9px; }
.mac-expansion > .q-expansion-item__container > .q-item { min-height:66px; padding:14px 22px;
  color:var(--text); font-size:18px; font-weight:600; }
.mac-expansion .nicegui-expansion-content { padding:4px 22px 22px; }
.q-dialog__backdrop { background:var(--overlay) !important; }
.mac-dialog { width:min(480px, 92vw); max-width:92vw; }
.q-notification { box-shadow:none; border-radius:7px; }
.q-tooltip { max-width:300px; background:var(--toolbar); color:var(--text);
  border:1px solid var(--line-strong); font-size:12px; }
.mac-locked { cursor:not-allowed; }
.mac-locked .disabled, .mac-locked [disabled] { cursor:not-allowed !important; }
body.mac-ended .nicegui-error-popup { display:none !important; }

@media (max-width:820px) {
  .mac-sidebar { width:58px; padding:24px 5px; }
  .mac-brand { margin:0 0 32px; justify-content:center; }
  .mac-wordmark, .mac-footer { display:none; }
  .mac-nav { min-width:0; padding:0; }
  .mac-nav .q-btn__content .block { display:none !important; }
  .mac-nav .q-icon { margin:0 !important; }
  .mac-page, .mac-page.is-wide { padding:30px 18px 48px; }
  .mac-title, .mac-page.is-wide .mac-title { font-size:29px; }
  .mac-home, .mac-two { grid-template-columns:minmax(0,1fr); }
  .mac-zone { min-height:390px; padding:30px 14px; }
  .mac-card { padding:18px 16px; }
  .mac-flow { padding:25px 22px; }
  .mac-segments { flex-wrap:nowrap; width:100%; }
  .mac-segments .q-btn { flex:1 1 0; min-width:0; padding:4px 6px; }
  .mac-item { gap:8px; }
  .mac-item-actions { flex-direction:column; }
}
"""


def style() -> str:
    variables = "".join(f"--{name}:{value};" for name, value in COLORS.items())
    # The overrides layer also wins over Quasar's !important utilities.
    return f"<style>@layer overrides {{ :root {{ {variables} }} {_RULES} }}</style>"
