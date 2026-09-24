// Clock
function updateClock() {
    const el = document.getElementById("clock");
    if (el) el.innerText = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// Monitoring Controls
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

// Telemetry Polling
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

        // Check if DB changed before re-rendering
        const currentIds = data.map(v => v.id || v.image_path).join(',');
        if (currentIds === cachedViolationIds) return;
        
        cachedViolationIds = currentIds;
        rawViolations = data;
        renderFeedAndGallery();
    } catch (err) {
        console.error("Failed to load violations:", err);
    }
}

// Small helper: student_track_id is NULL for object (phone) violations,
// since those aren't tied to a specific tracked person in this version --
// only render the tag when a real ID is present rather than showing
// "Student null".
function studentTag(v) {
    if (v.student_track_id === null || v.student_track_id === undefined) return '';
    return `<span class="student-tag">Student #${v.student_track_id}</span>`;
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

    // 1. Render Alerts Feed (Newest 15)
    rawViolations.slice(0, 15).forEach(v => {
        let typeClass = "";
        const uType = (v.violation_type || "").toUpperCase();
        if (uType.includes("GAZE")) typeClass = "gaze";
        else if (uType.includes("INTERACTION")) typeClass = "interaction";

        alerts.innerHTML += `
        <div class="alert-item ${typeClass}">
            <h4>${(v.violation_type || 'UNKNOWN').replace(/_/g, ' ')}</h4>
            <p>${v.timestamp || ''} ${studentTag(v)}</p>
        </div>
        `;
    });

    // 2. Render Filtered Gallery
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
                <small>${v.timestamp || ''} ${studentTag(v)}</small>
            </div>
        `;
        gallery.appendChild(card);
    });
}

// Lightbox Modal
function openModal(v) {
    document.getElementById("modalImg").src = `/static/${v.image_path}`;
    document.getElementById("modalType").innerText = (v.violation_type || 'INCIDENT').replace(/_/g, ' ');
    document.getElementById("modalTime").innerText = v.timestamp || '-';
    document.getElementById("modalConf").innerText = `${((v.confidence || 0.9) * 100).toFixed(1)}%`;

    const studentRow = document.getElementById("modalStudentRow");
    const studentVal = document.getElementById("modalStudent");
    if (v.student_track_id !== null && v.student_track_id !== undefined) {
        studentVal.innerText = `#${v.student_track_id}`;
        studentRow.style.display = "";
    } else {
        studentRow.style.display = "none";
    }

    document.getElementById("imageModal").style.display = "flex";
}

function closeModal(e) {
    document.getElementById("imageModal").style.display = "none";
}

// Filter Buttons
document.addEventListener("DOMContentLoaded", () => {
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

    setInterval(loadStats, 2000);
    setInterval(loadViolations, 2500);
});