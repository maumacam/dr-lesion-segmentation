const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const filenameEl = document.getElementById("filename");
const statusEl = document.getElementById("status");

const originalImg = document.getElementById("originalImg");
const segnetImg = document.getElementById("segnetImg");
const unetppImg = document.getElementById("unetppImg");

const segnetDice = document.getElementById("segnet-dice");
const segnetIoU = document.getElementById("segnet-iou");
const segnetSens = document.getElementById("segnet-sens");
const segnetSpec = document.getElementById("segnet-spec");

const unetppDice = document.getElementById("unetpp-dice");
const unetppIoU = document.getElementById("unetpp-iou");
const unetppSens = document.getElementById("unetpp-sens");
const unetppSpec = document.getElementById("unetpp-spec");

const showHeatmapsBtn = document.getElementById("showHeatmapsBtn");
const segnetHeatmap = document.getElementById("segnetHeatmap");
const unetppHeatmap = document.getElementById("unetppHeatmap");
const heatmapsContainer = document.getElementById("heatmapsContainer");

const insightsText = document.getElementById("insightsText");

heatmapsContainer.style.display = "none";

// ✅ Sanitize filename for safe URLs
function sanitizeFilename(name) {
  return name.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9._-]/g, "");
}

fileInput.addEventListener("change", () => {
  filenameEl.textContent = fileInput.files[0]?.name || "No file selected";
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return alert("Select a file first.");

  statusEl.textContent = "Uploading and processing...";
  statusEl.style.color = "blue";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/upload/", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const data = await response.json();
      alert("Upload failed: " + (data?.message || "Unknown error"));
      statusEl.textContent = "Failed.";
      statusEl.style.color = "red";
      return;
    }

    const data = await response.json();
    const safeFilename = sanitizeFilename(data.filename);

    // ✅ Show original uploaded image
    originalImg.src = URL.createObjectURL(file);

    // ✅ Segmentation results (check backend endpoints!)
    segnetImg.src = `http://127.0.0.1:8000/result/segnet/${safeFilename}`;
    unetppImg.src = `http://127.0.0.1:8000/result/unetpp/${safeFilename}`;

    // ✅ Metrics display
    segnetDice.textContent = data.metrics.segnet.dice.toFixed(2);
    segnetIoU.textContent = data.metrics.segnet.iou.toFixed(2);
    segnetSens.textContent = data.metrics.segnet.sensitivity.toFixed(2);
    segnetSpec.textContent = data.metrics.segnet.specificity.toFixed(2);

    unetppDice.textContent = data.metrics.unetpp.dice.toFixed(2);
    unetppIoU.textContent = data.metrics.unetpp.iou.toFixed(2);
    unetppSens.textContent = data.metrics.unetpp.sensitivity.toFixed(2);
    unetppSpec.textContent = data.metrics.unetpp.specificity.toFixed(2);

    insightsText.textContent =
      "✅ U-Net++ generally captures finer lesion details, SegNet is faster but less precise.";

    statusEl.textContent = "Processing complete!";
    statusEl.style.color = "green";
    filenameEl.textContent = safeFilename;

  } catch (err) {
    console.error("Error:", err);
    alert("Upload failed. Check console for details.");
    statusEl.textContent = "Failed.";
    statusEl.style.color = "red";
  }
});

// ✅ Heatmap toggle
showHeatmapsBtn.addEventListener("click", () => {
  if (heatmapsContainer.style.display === "none") {
    heatmapsContainer.style.display = "flex";
    const safeFilename = sanitizeFilename(filenameEl.textContent);
    segnetHeatmap.src = `http://127.0.0.1:8000/result/heatmap/segnet/${safeFilename}`;
    unetppHeatmap.src = `http://127.0.0.1:8000/result/heatmap/unetpp/${safeFilename}`;
  } else {
    heatmapsContainer.style.display = "none";
  }
});
