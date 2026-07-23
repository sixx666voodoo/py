# Forza Horizon 6 Flexbar Companion

Starter Flexbar plugin concept for Forza Horizon 6. The first version focuses on
useful shortcut pages and leaves clean integration points for telemetry once the
final game data-output behavior is known.

## Planned pages

- **Cruise**: map, rewind, camera, radio, photo mode, and telemetry toggle.
- **Race**: rewind, look back, camera, pause, clip capture, and telemetry HUD.
- **Drift / Touge**: handbrake, telemetry, reset car, photo, clip capture, and tuning shortcut.
- **Photo / Creator**: photo mode, screenshot, hide UI, OBS, captures folder, and back button.
- **Telemetry HUD**: speed, gear, RPM, skill multiplier, and connection state.

## Development notes

The backend currently runs as a static shortcut companion and exposes a small
`updateTelemetry` pathway for future live data. When Forza Horizon 6 telemetry
is confirmed, wire the telemetry listener into `backend/index.js` and forward
normalized speed/RPM/gear data to the dynamic key renderer.

## Suggested local workflow

```bash
npm install -g @eniactech/flexcli
flexcli link ./flexbar-forza-horizon-6-plugin
flexcli restart
```

If your FlexCLI package is published under a different npm scope, use the scope
that matches your installed FlexDesigner SDK version.
