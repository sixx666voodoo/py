"use strict";

/**
 * Starter backend for a Flexbar Forza Horizon 6 companion plugin.
 *
 * The implementation is intentionally defensive: it works as a static shortcut
 * companion today and keeps telemetry/game-state integration behind small
 * adapter functions that can be wired to the final Forza Horizon 6 telemetry
 * interface when it is available.
 */

const { plugin, logger } = require("@eniactech/flexdesigner-sdk");

const DEFAULT_PROFILE = {
  activeMode: "cruise",
  telemetry: {
    connected: false,
    speedMph: 0,
    rpm: 0,
    gear: "N",
    skillMultiplier: "1.0x",
  },
};

const MODES = {
  cruise: {
    title: "Cruise",
    accent: "#35d0ff",
    actions: ["Map", "Rewind", "Camera", "Radio", "Photo", "Telemetry"],
  },
  race: {
    title: "Race",
    accent: "#ff3b30",
    actions: ["Rewind", "Look Back", "Camera", "Pause", "Clip", "Telemetry"],
  },
  drift: {
    title: "Drift",
    accent: "#b56cff",
    actions: ["Handbrake", "Telemetry", "Reset", "Photo", "Clip", "Tune"],
  },
  photo: {
    title: "Photo",
    accent: "#ffd60a",
    actions: ["Photo Mode", "Screenshot", "Hide UI", "OBS", "Captures", "Back"],
  },
};

const state = structuredCloneSafe(DEFAULT_PROFILE);

plugin.on?.("plugin.alive", () => {
  logger.info?.("Forza Horizon 6 Companion started");
  renderActiveMode();
});

plugin.on?.("plugin.dead", () => {
  logger.info?.("Forza Horizon 6 Companion stopped");
});

plugin.on?.("plugin.data", (payload) => {
  if (!payload || typeof payload !== "object") {
    return;
  }

  if (payload.action === "setMode" && MODES[payload.mode]) {
    state.activeMode = payload.mode;
    renderActiveMode();
    return;
  }

  if (payload.action === "telemetry") {
    updateTelemetry(payload.telemetry || {});
  }
});

function renderActiveMode() {
  const mode = MODES[state.activeMode] || MODES.cruise;
  logger.info?.(`Rendering ${mode.title} mode with actions: ${mode.actions.join(", ")}`);

  if (!plugin.dynamickey?.clear || !plugin.dynamickey?.add) {
    return;
  }

  plugin.dynamickey.clear();
  plugin.dynamickey.add({
    id: "fh6.mode.header",
    title: `${mode.title} Mode`,
    subtitle: state.telemetry.connected ? "Telemetry online" : "Shortcut page",
    color: mode.accent,
  });

  for (const action of mode.actions) {
    plugin.dynamickey.add({
      id: `fh6.action.${slug(action)}`,
      title: action,
      color: mode.accent,
    });
  }

  renderTelemetryHud();
}

function updateTelemetry(nextTelemetry) {
  state.telemetry = {
    ...state.telemetry,
    ...nextTelemetry,
    connected: true,
  };
  renderTelemetryHud();
}

function renderTelemetryHud() {
  if (!plugin.dynamickey?.add && !plugin.dynamickey?.update) {
    return;
  }

  const telemetry = state.telemetry;
  const payload = {
    id: "fh6.telemetry.hud",
    title: `${Math.round(telemetry.speedMph)} mph`,
    subtitle: `Gear ${telemetry.gear} · ${Math.round(telemetry.rpm)} rpm · ${telemetry.skillMultiplier}`,
    color: telemetry.connected ? "#32d74b" : "#8e8e93",
  };

  if (plugin.dynamickey?.update) {
    plugin.dynamickey.update(payload);
  } else {
    plugin.dynamickey.add(payload);
  }
}

function slug(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function structuredCloneSafe(value) {
  return JSON.parse(JSON.stringify(value));
}


module.exports = {
  MODES,
  DEFAULT_PROFILE,
  slug,
};
