const POLL_MS = 300;
const MAX_LOG_LINES = 25;

const leftLight = document.getElementById("trafficLightLeft");
const rightLight = document.getElementById("trafficLightRight");
const safetyText = document.getElementById("safetyText");
const roadText = document.getElementById("roadText");
const v2iText = document.getElementById("v2iText");

const safetyCard = document.getElementById("safetyCard");
const roadCard = document.getElementById("roadCard");
const v2iCard = document.getElementById("v2iCard");

// 🚨 NEW: References for the new cards
const multiEmergencyCard = document.getElementById("multiEmergencyCard");
const festivalStatusCard = document.getElementById("festivalStatusCard");

const trafficCountA = document.getElementById("trafficCountA");
const trafficCountB = document.getElementById("trafficCountB");
const countdownTimer = document.getElementById("countdownTimer");
const logBox = document.getElementById("logBox");
const uptimeDisplay = document.getElementById("uptime");
const greenCorridorText = document.getElementById("greenCorridorText");
const greenCorridorItem = document.getElementById("greenCorridorItem");
const festivalToggle = document.getElementById("festivalToggle");

const demoOverlay = document.getElementById("demoOverlay");
const demoOverlayDismiss = document.getElementById("demoOverlayDismiss");
const commandInput = document.getElementById("commandInput");
const commandRun = document.getElementById("commandRun");
const commandOutput = document.getElementById("commandOutput");
const eventsBox = document.getElementById("eventsBox");
const eventsRefresh = document.getElementById("eventsRefresh");
const videoError = document.getElementById("videoError");
const videoErrorDetail = document.getElementById("videoErrorDetail");
const liveBadge = document.getElementById("liveBadge");
const liveFeed = document.getElementById("liveFeed");
const statusBadges = document.getElementById("statusBadges");
const videoLoading = document.getElementById("videoLoading");
const eventFilter = document.getElementById("eventFilter");

let lastLog = "";
let statusFailCount = 0;
let lastSosState = false;
let lastAccidentState = false;
let lastAmbulanceState = false;
let feedLoaded = false;

if (demoOverlay) {
  const dismissed = localStorage.getItem("sutra_demo_dismissed");
  if (dismissed === "1") demoOverlay.classList.add("hidden");
}
demoOverlayDismiss?.addEventListener("click", () => {
  demoOverlay?.classList.add("hidden");
  localStorage.setItem("sutra_demo_dismissed", "1");
});
document.getElementById("helpBtn")?.addEventListener("click", () => {
  demoOverlay?.classList.remove("hidden");
});

function playAlertSound(type) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    gain.gain.value = 0.15;
    if (type === "sos") {
      osc.frequency.value = 880; osc.type = "sine"; osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.3);
    } else if (type === "accident") {
      osc.frequency.value = 400; osc.type = "square"; osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.5);
    } else if (type === "ambulance") {
      osc.frequency.value = 600; osc.type = "sine"; osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.2);
    }
  } catch (e) {}
}

let uptimeSeconds = 0;
setInterval(() => {
  uptimeSeconds++;
  const h = Math.floor(uptimeSeconds / 3600).toString().padStart(2, "0");
  const m = Math.floor((uptimeSeconds % 3600) / 60).toString().padStart(2, "0");
  const s = (uptimeSeconds % 60).toString().padStart(2, "0");
  if (uptimeDisplay) uptimeDisplay.textContent = `${h}:${m}:${s}`;
}, 1000);

function updateTrafficLight(lightContainer, activeColor) {
  if (!lightContainer) return;
  const lights = lightContainer.querySelectorAll(".light");
  lights.forEach((l) => l.classList.remove("active"));
  const active = lightContainer.querySelector(`.light[data-color="${activeColor}"]`);
  if (active) active.classList.add("active");
}

function resetCardClasses(card) {
  if (!card) return;
  card.classList.remove("safe", "warning", "alert");
}

function appendLog(line) {
  if (!line || line === lastLog) return;
  lastLog = line;
  const now = new Date().toLocaleTimeString("en-US", { hour12: false });
  const entry = document.createElement("div");

  if (line.includes("EMERGENCY") || line.includes("SOS") || line.includes("Closed") || line.includes("SEVERE") || line.includes("COLLISION") || line.includes("AMBULANCE") || line.includes("FIRE") || line.includes("CRASH")) {
    entry.style.color = "var(--danger)";
  } else if (line.includes("Festival") || line.includes("GREEN CORRIDOR")) {
    entry.style.color = "var(--success)";
  } else if (line.includes("warning") || line.includes("Cattle") || line.includes("Animal") || line.includes("Force-Off") || line.includes("Truncating") || line.includes("DISPATCHED")) {
    entry.style.color = "var(--warning)";
  } else {
    entry.style.color = "var(--success)";
  }

  entry.innerHTML = `> <span style="color:var(--text-secondary)">[${now}]</span> ${line}`;
  logBox.appendChild(entry);
  while (logBox.children.length > MAX_LOG_LINES) logBox.removeChild(logBox.firstChild);
  logBox.scrollTop = logBox.scrollHeight;
}

function updateStatusBadges(data) {
  if (!statusBadges) return;
  statusBadges.innerHTML = "";
  const badges = [];
  if (data.demo_mode) badges.push({ c: "warn", t: "Demo" });
  if (data.camera_error && data.camera_switch_countdown > 0) badges.push({ c: "warn", t: `Camera: ${data.camera_switch_countdown}s` });
  else if (data.using_video_fallback) badges.push({ c: "warn", t: "Video" });
  else if (data.feed_available) badges.push({ c: "ok", t: "Feed" });
  else badges.push({ c: "err", t: "No Feed" });
  if (!data.audio_available) badges.push({ c: "warn", t: "No Audio" });
  badges.forEach((b) => {
    const span = document.createElement("span");
    span.className = `status-badge ${b.c}`;
    span.textContent = b.t;
    statusBadges.appendChild(span);
  });
}

function updateCountdownStyle(el, countdown) {
  if (!el) return;
  const isPaused = countdown === 99;
  const isLow = !isPaused && countdown <= 3 && countdown > 0;
  el.dataset.paused = isPaused ? "true" : "false";
  el.dataset.low = isLow ? "true" : "false";
  el.classList.toggle("countdown-paused", isPaused);
  el.classList.toggle("countdown-low", isLow);
}

async function refreshEvents() {
  try {
    const filter = eventFilter?.value || "all";
    const url = filter === "all" ? "/events" : `/events?type=${encodeURIComponent(filter)}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return;
    const events = await res.json();
    if (!eventsBox) return;
    eventsBox.innerHTML = events.slice(-20).reverse().map((e) =>
      `<div class="event"><span class="event-type">[${e.type}]</span>${e.message}</div>`
    ).join("") || "<div class='event'>No events</div>";
  } catch (e) {
    eventsBox.innerHTML = "<div class='event'>Failed to load</div>";
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/status", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusFailCount = 0;

    updateTrafficLight(leftLight, data.traffic_light_a);
    updateTrafficLight(rightLight, data.traffic_light_b);

    if (safetyText) safetyText.textContent = data.safety;
    if (roadText) roadText.textContent = data.road;
    if (v2iText) v2iText.textContent = data.v2i_status;
    if (trafficCountA) trafficCountA.textContent = data.traffic_count_a;
    if (trafficCountB) trafficCountB.textContent = data.traffic_count_b;

    if (countdownTimer) {
      countdownTimer.textContent = data.countdown === 99 ? "PAUSED" : data.countdown + "s";
      updateCountdownStyle(countdownTimer, data.countdown);
    }

    if (greenCorridorText) greenCorridorText.textContent = data.green_corridor_active ? "ACTIVE" : "Inactive";
    if (greenCorridorItem) greenCorridorItem.classList.toggle("green-corridor-active", !!data.green_corridor_active);
    if (festivalToggle) {
      festivalToggle.textContent = data.festival_mode ? "ON" : "OFF";
      festivalToggle.classList.toggle("active", !!data.festival_mode);
    }

    updateStatusBadges(data);

    // 🚨 NEW: Auto-create/show Multi-Emergency Tab if BOTH exist
    if (multiEmergencyCard) {
      if (data.green_corridor_active && data.road.includes("CRASH")) {
        multiEmergencyCard.style.display = "flex";
      } else {
        multiEmergencyCard.style.display = "none";
      }
    }

    // 🚨 NEW: Auto-create/show separate Festival Tab
    if (festivalStatusCard) {
      if (data.festival_mode) {
        festivalStatusCard.style.display = "flex";
      } else {
        festivalStatusCard.style.display = "none";
      }
    }

    if (videoLoading && !feedLoaded) {
      if (data.feed_available) {
        videoLoading.classList.add("hidden");
        feedLoaded = true;
      } else if (statusFailCount === 0) {
        setTimeout(() => {
          if (!feedLoaded && videoLoading) videoLoading.classList.add("hidden");
        }, 5000);
      }
    }

    const sosActive = data.safety.includes("SOS") || data.safety.includes("HELP");
    const accidentActive = data.road.includes("CRASH");
    const ambulanceActive = !!data.green_corridor_active;
    if (sosActive && !lastSosState) playAlertSound("sos");
    if (accidentActive && !lastAccidentState) playAlertSound("accident");
    if (ambulanceActive && !lastAmbulanceState) playAlertSound("ambulance");
    lastSosState = sosActive; lastAccidentState = accidentActive; lastAmbulanceState = ambulanceActive;

    if (videoError && videoErrorDetail) {
      if (!data.feed_available) {
        videoError.style.display = "flex";
        let detail = "Feed unavailable.";
        if (data.camera_error && data.camera_switch_countdown > 0) detail = `Camera not working. Switching to video in ${data.camera_switch_countdown}s...`;
        else if (data.using_video_fallback) detail = "Using video fallback.";
        videoErrorDetail.textContent = detail;
      } else {
        videoError.style.display = "none";
      }
    }
    if (liveBadge) liveBadge.classList.toggle("offline", !data.feed_available);

    resetCardClasses(safetyCard); resetCardClasses(roadCard); resetCardClasses(v2iCard);

    if (data.safety.includes("SOS") || data.safety.includes("HELP")) safetyCard.classList.add("alert");
    else safetyCard.classList.add("safe");

    if (data.road.includes("CRASH")) roadCard.classList.add("alert");
    else if (data.road.includes("DISPATCHED")) roadCard.classList.add("warning");
    else roadCard.classList.add("safe");

    if (data.v2i_status.includes("OVERRIDE") || data.v2i_status.includes("CRASH")) v2iCard.classList.add("alert");
    else if (data.v2i_status.includes("FORCE") || data.v2i_status.includes("EXTENSION") || data.v2i_status.includes("BUS")) v2iCard.classList.add("warning");
    else v2iCard.classList.add("safe");

    appendLog(data.ai_log);
  } catch (error) {
    statusFailCount++;
    console.warn("S.U.T.R.A Status Sync Drop:", error.message);
    if (statusBadges) statusBadges.innerHTML = '<span class="status-badge err">Status Error</span>';
    if (videoError && videoErrorDetail && statusFailCount > 2) {
      videoError.style.display = "flex";
      videoErrorDetail.textContent = "Cannot connect to server. Is the app running?";
    }
    if (liveBadge) liveBadge.classList.add("offline");
  }
}

async function runCommand() {
  const cmd = commandInput?.value?.trim() || "";
  if (!cmd) return;
  try {
    const res = await fetch("/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd }),
    });
    const data = await res.json();
    if (commandOutput) {
      commandOutput.textContent = data.output || (data.ok ? "OK" : "Error");
      commandOutput.style.color = data.ok ? "var(--success)" : "var(--danger)";
    }
    if (data.ok && (cmd === "/demo" || cmd === "/demomode")) refreshStatus();
    if (data.ok && (cmd === "/events" || cmd === "/event")) refreshEvents();
    if (data.ok && cmd === "/reload") refreshStatus();
  } catch (e) {
    if (commandOutput) {
      commandOutput.textContent = "Command failed: " + e.message;
      commandOutput.style.color = "var(--danger)";
    }
  }
}

setInterval(refreshStatus, POLL_MS);
refreshStatus();
refreshEvents();

commandRun?.addEventListener("click", runCommand);
commandInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") runCommand(); });
eventsRefresh?.addEventListener("click", refreshEvents);
eventFilter?.addEventListener("change", refreshEvents);

festivalToggle?.addEventListener("click", async () => {
  try {
    const res = await fetch("/festival_mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !festivalToggle.classList.contains("active") }),
    });
    if (res.ok) {
      const data = await res.json();
      festivalToggle.textContent = data.festival_mode ? "ON" : "OFF";
      festivalToggle.classList.toggle("active", !!data.festival_mode);
    }
  } catch (e) { console.warn("Festival mode toggle failed:", e.message); }
});

liveFeed?.addEventListener("error", () => {
  if (videoError) videoError.style.display = "flex";
  if (videoErrorDetail) videoErrorDetail.textContent = "Video feed failed to load.";
  if (liveBadge) liveBadge.classList.add("offline");
});