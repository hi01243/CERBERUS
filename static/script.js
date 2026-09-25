// Clock
function updateClock() {
    const el = document.getElementById("clock");
    if (el) el.innerText = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// ==========================================
// CAMERA MANAGEMENT
// ==========================================

let cachedCameraSignature = "";

async function loadCameras() {
    try {
        const res = await fetch('/cameras');
        if (!res.ok) return;
        const cameras = await res.json();

        // Only re-render the grid if something actually changed (a camera
        // added/removed, or a status flip). Re-rendering on every poll
        // tick would recreate the <img> elements for active streams,
        // which restarts their MJPEG connection and causes flicker.
        const signature = cameras.map(c => `${c.id}:${c.status}:${c.label}`).join('|');
        if (signature === cachedCameraSignature) return;
        cachedCameraSignature = signature;

        renderCameraGrid(cameras);
        document.getElementById("cameraCountTag").innerText = cameras.length;
    } catch (err) {
        console.error("Failed to load cameras:", err);
    }
}

function renderCameraGrid(cameras) {
    const grid = document.getElementById("cameraGrid");
    if (!grid) return;

    if (cameras.length === 0) {
        grid.innerHTML = '<div class="empty-state">No cameras registered yet. Scan for a local camera or add one by IP above.</div>';
        return;
    }

    grid.innerHTML = cameras.map(cam => {
        const isActive = cam.status === "active";
        const isPaused = cam.status === "paused";

        const mediaHtml = isActive
            ? `<img src="/video_feed/${cam.id}" class="video-feed" alt="Camera ${cam.label}">`
            : `<div class="camera-placeholder">
                   <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                       <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                   </svg>
                   <p>${isPaused ? "Paused" : "Not in use"}</p>
               </div>`;

        const actionBtn = isActive
            ? `<button class="tile-btn" onclick="pauseCamera('${cam.id}')">⏸ Pause</button>`
            : `<button class="tile-btn tile-btn-primary" onclick="useCamera('${cam.id}')">▶ Use</button>`;

        return `
        <div class="camera-tile">
            <span class="camera-tile-label">${cam.label}</span>
            <span class="camera-status-badge status-${cam.status}">${cam.status.toUpperCase()}</span>
            ${mediaHtml}
            <div class="camera-tile-controls">
                ${actionBtn}
                <button class="tile-btn tile-btn-danger" onclick="removeCamera('${cam.id}')">✕ Remove</button>
            </div>
        </div>`;
    }).join('');
}

async function scanCameras() {
    const statusEl = document.getElementById("scanStatus");
    const resultsEl = document.getElementById("scanResults");
    statusEl.innerText = "Scanning... (this can take a few seconds)";
    resultsEl.innerHTML = "";

    try {
        const res = await fetch('/cameras/scan', { method: 'POST' });
        const data = await res.json();
        const found = data.found_indices || [];

        if (found.length === 0) {
            statusEl.innerText = "No new local cameras found.";
            return;
        }

        statusEl.innerText = `Found ${found.length} camera(s):`;
        resultsEl.innerHTML = found.map(idx => `
            <button class="tile-btn tile-btn-primary scan-add-btn" onclick="addCamera('${idx}', 'Local Camera ${idx}')">
                + Add Local Camera ${idx}
            </button>
        `).join('');
    } catch (err) {
        console.error("Scan failed:", err);
        statusEl.innerText = "Scan failed — check the console.";
    }
}

async function addCamera(source, label) {
    try {
        const res = await fetch('/cameras/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source, label })
        });
        const data = await res.json();
        if (!res.ok) {
            alert(data.message || "Failed to add camera.");
            return;
        }
        document.getElementById("scanResults").innerHTML = "";
        document.getElementById("scanStatus").innerText = "";
        cachedCameraSignature = ""; // force a re-render
        loadCameras();
    } catch (err) {
        console.error("Add camera failed:", err);
        alert("Failed to add camera — check the console.");
    }
}

async function addIpCamera() {
    const urlInput = document.getElementById("ipCameraUrl");
    const labelInput = document.getElementById("ipCameraLabel");
    const statusEl = document.getElementById("addCameraStatus");

    const url = urlInput.value.trim();
    if (!url) {
        statusEl.innerText = "Enter a camera URL first.";
        return;
    }

    statusEl.innerText = "Adding...";
    try {
        const res = await fetch('/cameras/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: url, label: labelInput.value.trim() })
        });
        const data = await res.json();
        if (!res.ok) {
            statusEl.innerText = data.message || "Failed to add camera.";
            return;
        }
        statusEl.innerText = "Camera added — click 'Use' on its tile to start it.";
        urlInput.value = "";
        labelInput.value = "";
        cachedCameraSignature = "";
        loadCameras();
    } catch (err) {
        console.error("Add IP camera failed:", err);
        statusEl.innerText = "Failed — check the console.";
    }
}

async function useCamera(camId) {
    try {
        const res = await fetch(`/cameras/${camId}/activate`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) {
            alert(data.message || "Could not activate this camera. Is it already in use by another app?");
            return;
        }
        cachedCameraSignature = "";
        loadCameras();
    } catch (err) {
        console.error("Activate failed:", err);
    }
}

async function pauseCamera(camId) {
    try {
        await fetch(`/cameras/${camId}/pause`, { method: 'POST' });
        cachedCameraSignature = "";
        loadCameras();
    } catch (err) {
        console.error("Pause failed:", err);
    }
}

async function removeCamera(camId) {
    if (!confirm(`Remove camera "${camId}"? This stops its stream and forgets it — you'll need to Scan/Add again to bring it back.`)) {
        return;
    }
    try {
        await fetch(`/cameras/${camId}/remove`, { method: 'POST' });
        cachedCameraSignature = "";
        loadCameras();
    } catch (err) {
        console.error("Remove failed:", err);
    }
}

// ==========================================
// MONITORING CONTROLS (session-wide arm/disarm)
// ==========================================

async function startMonitoring() {
    try {
        await fetch('/start_monitoring', { method: 'POST' });
        loadStats();
    } catch (err) {
        console.error("Start failed:", err);
    }
}

async function stopMonitoring() {
    try {
        await fetch('/stop_monitoring', { method: 'POST' });
        loadStats();
    } catch (err) {
        console.error("Stop failed:", err);
    }
}

// Destructive: wipes DB rows + evidence photos for the whole session,
// across every camera.
async function resetSession() {
    if (!confirm(
        "This will permanently delete ALL logged violations and evidence " +
        "photos for this session, across every camera. This cannot be undone.\n\n" +
        "Continue?"
    )) {
        return;
    }

    try {
        const res = await fetch('/reset_session', { method: 'POST' });
        const data = await res.json();

        if (!res.ok) {
            alert(data.message || "Reset failed. Make sure monitoring is paused first.");
            return;
        }

        cachedViolationIds = "";
        rawViolations = [];
        renderFeedAndGallery();
        loadStats();
    } catch (err) {
        console.error("Reset failed:", err);
        alert("Reset failed — check the browser console.");
    }
}

// ==========================================
// TELEMETRY / VIOLATIONS POLLING
// ==========================================

async function loadStats() {
    try {
        const res = await fetch('/system_stats');
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("totalAlerts").innerText = data.total_violations ?? 0;
        document.getElementById("phoneAlerts").innerText = data.phone ?? 0;
        document.getElementById("interactionAlerts").innerText = data.interaction ?? 0;
        document.getElementById("gazeAlerts").innerText = data.gaze ?? 0;

        const statusLabel = document.getElementById("monitorStatus");
        const statusDot = document.getElementById("statusDot");

        if (data.monitoring) {
            statusLabel.innerText = "ACTIVE";
            statusLabel.className = "stat-value online";
            statusDot.className = "indicator-dot online";
        } else {
            statusLabel.innerText = "PAUSED";
            statusLabel.className = "stat-value offline";
            statusDot.className = "indicator-dot offline";
        }
    } catch (err) {
        console.error("Failed to load telemetry:", err);
    }
}

let cachedViolationIds = "";
let rawViolations = [];
let activeFilter = "ALL";

async function loadViolations() {
    try {
        const res = await fetch('/get_violations');
        if (!res.ok) return;
        const data = await res.json();

        const currentIds = data.map(v => v.id || v.image_path).join(',');
        if (currentIds === cachedViolationIds) return;

        cachedViolationIds = currentIds;
        rawViolations = data;
        renderFeedAndGallery();
    } catch (err) {
        console.error("Failed to load violations:", err);
    }
}

function studentTag(v) {
    if (v.student_track_id === null || v.student_track_id === undefined) return '';
    return `<span class="student-tag">Student #${v.student_track_id}</span>`;
}

function cameraTag(v) {
    if (!v.camera_id) return '';
    return `<span class="camera-tag">${v.camera_id}</span>`;
}

function renderFeedAndGallery() {
    const alerts = document.getElementById('alertsContainer');
    const gallery = document.getElementById('galleryGrid');
    const countLabel = document.getElementById('alertCount');

    if (!alerts || !gallery) return;

    countLabel.innerText = `${rawViolations.length} events`;
    alerts.innerHTML = '';
    gallery.innerHTML = '';

    if (rawViolations.length === 0) {
        alerts.innerHTML = '<div class="empty-state">No incidents logged.</div>';
        gallery.innerHTML = '<div class="empty-state">No proof frames captured in this session yet.</div>';
        return;
    }

    rawViolations.slice(0, 15).forEach(v => {
        let typeClass = "";
        const uType = (v.violation_type || "").toUpperCase();
        if (uType.includes("GAZE")) typeClass = "gaze";
        else if (uType.includes("INTERACTION")) typeClass = "interaction";

        alerts.innerHTML += `
        <div class="alert-item ${typeClass}">
            <h4>${(v.violation_type || 'UNKNOWN').replace(/_/g, ' ')}</h4>
            <p>${v.timestamp || ''} ${cameraTag(v)} ${studentTag(v)}</p>
        </div>
        `;
    });

    const filtered = rawViolations.filter(v => {
        if (activeFilter === "ALL") return true;
        return (v.violation_type || "").toUpperCase().includes(activeFilter);
    });

    if (filtered.length === 0) {
        gallery.innerHTML = '<div class="empty-state">No evidence under this filter.</div>';
        return;
    }

    filtered.forEach(v => {
        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.onclick = () => openModal(v);

        card.innerHTML = `
            <img src="/static/${v.image_path}?t=${Date.now()}" loading="lazy">
            <div class="gallery-info">
                <strong>${(v.violation_type || 'UNKNOWN').replace(/_/g, ' ')}</strong>
                <small>${v.timestamp || ''} ${cameraTag(v)} ${studentTag(v)}</small>
            </div>
        `;
        gallery.appendChild(card);
    });
}

function openModal(v) {
    document.getElementById("modalImg").src = `/static/${v.image_path}`;
    document.getElementById("modalType").innerText = (v.violation_type || 'INCIDENT').replace(/_/g, ' ');
    document.getElementById("modalTime").innerText = v.timestamp || '-';
    document.getElementById("modalConf").innerText = `${((v.confidence || 0.9) * 100).toFixed(1)}%`;

    const cameraRow = document.getElementById("modalCameraRow");
    if (v.camera_id) {
        document.getElementById("modalCamera").innerText = v.camera_id;
        cameraRow.style.display = "";
    } else {
        cameraRow.style.display = "none";
    }

    const studentRow = document.getElementById("modalStudentRow");
    if (v.student_track_id !== null && v.student_track_id !== undefined) {
        document.getElementById("modalStudent").innerText = `#${v.student_track_id}`;
        studentRow.style.display = "";
    } else {
        studentRow.style.display = "none";
    }

    document.getElementById("imageModal").style.display = "flex";
}

function closeModal(e) {
    document.getElementById("imageModal").style.display = "none";
}

// Filter Buttons + Initial Load
document.addEventListener("DOMContentLoaded", () => {
    loadCameras();
    loadStats();
    loadViolations();

    document.querySelectorAll(".filter-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            activeFilter = e.target.dataset.filter;
            renderFeedAndGallery();
        });
    });

    setInterval(loadCameras, 5000);
    setInterval(loadStats, 2000);
    setInterval(loadViolations, 2500);
});