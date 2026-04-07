/**
 * DeepFake Detector — Frontend Logic
 * Handles file upload, drag-and-drop, API communication, and result display.
 */

// ─── Configuration ─────────────────────────────────────────────────────────
// Change this to your deployed backend URL when deploying
const API_URL = "http://localhost:8000";

// ─── DOM Elements ──────────────────────────────────────────────────────────
const uploadZone = document.getElementById("upload-zone");
const uploadContent = document.getElementById("upload-content");
const videoPreview = document.getElementById("video-preview");
const previewPlayer = document.getElementById("preview-player");
const fileInfo = document.getElementById("file-info");
const fileInput = document.getElementById("file-input");
const btnChange = document.getElementById("btn-change");
const btnAnalyze = document.getElementById("btn-analyze");
const btnText = document.querySelector(".btn-text");
const btnLoader = document.getElementById("btn-loader");
const resultCard = document.getElementById("result-card");
const resultIcon = document.getElementById("result-icon");
const resultVerdict = document.getElementById("result-verdict");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBarFill = document.getElementById("confidence-bar-fill");
const resultDescription = document.getElementById("result-description");
const btnReset = document.getElementById("btn-reset");
const errorCard = document.getElementById("error-card");
const errorMessage = document.getElementById("error-message");
const btnErrorReset = document.getElementById("btn-error-reset");
const mainCard = document.getElementById("main-card");

// ─── State ─────────────────────────────────────────────────────────────────
let selectedFile = null;

// ─── Allowed types & max size ──────────────────────────────────────────────
const ALLOWED_TYPES = [
    "video/mp4",
    "video/avi",
    "video/quicktime",        // .mov
    "video/x-matroska",       // .mkv
    "video/webm",
    "video/x-msvideo",        // .avi alternate
];
const ALLOWED_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"];
const MAX_SIZE_MB = 100;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

// ─── Helpers ───────────────────────────────────────────────────────────────
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function getFileExtension(name) {
    return name.split(".").pop().toLowerCase();
}

function isValidFile(file) {
    const ext = getFileExtension(file.name);
    return ALLOWED_EXTENSIONS.includes(ext);
}

// ─── File Selection ────────────────────────────────────────────────────────
function handleFile(file) {
    // Validate type
    if (!isValidFile(file)) {
        showError(
            "Invalid File Type",
            `"${file.name}" is not a supported video format. Please upload MP4, AVI, MOV, MKV, or WebM.`
        );
        return;
    }

    // Validate size
    if (file.size > MAX_SIZE_BYTES) {
        showError(
            "File Too Large",
            `"${file.name}" is ${formatFileSize(file.size)}. Maximum allowed size is ${MAX_SIZE_MB}MB.`
        );
        return;
    }

    selectedFile = file;

    // Show preview
    const url = URL.createObjectURL(file);
    previewPlayer.src = url;
    previewPlayer.onloadeddata = () => previewPlayer.play().catch(() => {});

    fileInfo.textContent = `${file.name} • ${formatFileSize(file.size)}`;

    uploadContent.style.display = "none";
    videoPreview.style.display = "block";
    btnAnalyze.disabled = false;

    // Hide any previous results / errors
    resultCard.style.display = "none";
    errorCard.style.display = "none";
}

// ─── Click to Upload ───────────────────────────────────────────────────────
uploadZone.addEventListener("click", (e) => {
    if (e.target === btnChange || e.target.closest("#btn-change")) return;
    fileInput.click();
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

btnChange.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
    fileInput.click();
});

// ─── Drag and Drop ─────────────────────────────────────────────────────────
uploadZone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// ─── Analyze ───────────────────────────────────────────────────────────────
btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Show loading state
    btnText.style.display = "none";
    btnLoader.style.display = "flex";
    btnAnalyze.disabled = true;

    try {
        const formData = new FormData();
        formData.append("video", selectedFile);

        const response = await fetch(`${API_URL}/api/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(
                errorData.detail || `Server error (${response.status})`
            );
        }

        const data = await response.json();
        showResult(data);
    } catch (error) {
        let msg = error.message;
        if (error.name === "TypeError" && msg.includes("fetch")) {
            msg = "Cannot reach the API server. Please make sure the backend is running.";
        }
        showError("Analysis Failed", msg);
    } finally {
        // Reset button state
        btnText.style.display = "flex";
        btnLoader.style.display = "none";
        btnAnalyze.disabled = false;
    }
});

// ─── Show Result ───────────────────────────────────────────────────────────
function showResult(data) {
    const isReal = data.result === "REAL";
    const confidence = data.confidence;

    // Icon
    resultIcon.className = `result-icon ${isReal ? "real" : "fake"}`;
    resultIcon.innerHTML = isReal ? "✅" : "🚨";

    // Verdict
    resultVerdict.className = `result-verdict ${isReal ? "real" : "fake"}`;
    resultVerdict.textContent = isReal
        ? "Authentic Video"
        : "DeepFake Detected";

    // Confidence
    confidenceValue.textContent = `${confidence}%`;

    // Confidence bar — animate after a short delay
    confidenceBarFill.className = `confidence-bar-fill ${isReal ? "real" : "fake"}`;
    confidenceBarFill.style.width = "0%";
    requestAnimationFrame(() => {
        setTimeout(() => {
            confidenceBarFill.style.width = `${confidence}%`;
        }, 100);
    });

    // Description
    if (isReal) {
        resultDescription.textContent =
            `Our AI model has analyzed the video and determined it is likely authentic with ${confidence}% confidence. ` +
            `No significant signs of face manipulation, warping, or synthetic artifacts were detected.`;
    } else {
        resultDescription.textContent =
            `Our AI model has detected signs of manipulation in this video with ${confidence}% confidence. ` +
            `The analysis found patterns consistent with deepfake generation techniques such as face swapping or reenactment.`;
    }

    // Show result, hide upload
    mainCard.style.display = "none";
    errorCard.style.display = "none";
    resultCard.style.display = "block";
}

// ─── Show Error ────────────────────────────────────────────────────────────
function showError(title, message) {
    document.getElementById("error-title").textContent = title;
    errorMessage.textContent = message;
    errorCard.style.display = "block";
    resultCard.style.display = "none";
}

// ─── Reset ─────────────────────────────────────────────────────────────────
function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewPlayer.src = "";
    uploadContent.style.display = "block";
    videoPreview.style.display = "none";
    btnAnalyze.disabled = true;
}

function fullReset() {
    resetUpload();
    resultCard.style.display = "none";
    errorCard.style.display = "none";
    mainCard.style.display = "block";
}

btnReset.addEventListener("click", fullReset);
btnErrorReset.addEventListener("click", fullReset);
