// =====================================================================
// LUMINASCAN AI - LOGIC
// =====================================================================

let uploadedData = null;

// =====================================================================
// INITIALIZATION
// =====================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    // Initialize buttons
    if (document.getElementById('applyEnhancement'))
        document.getElementById('applyEnhancement').addEventListener('click', runEnhancement);
    if (document.getElementById('applyNoise'))
        document.getElementById('applyNoise').addEventListener('click', runNoise);
    if (document.getElementById('runSegmentation'))
        document.getElementById('runSegmentation').addEventListener('click', runSegmentation);
    if (document.getElementById('computeHistograms'))
        document.getElementById('computeHistograms').addEventListener('click', runHistograms);
    if (document.getElementById('runPipelines'))
        document.getElementById('runPipelines').addEventListener('click', runPipelines);
    if (document.getElementById('analyzeRAG'))
        document.getElementById('analyzeRAG').addEventListener('click', runRAG);
    if (document.getElementById('askFollowup'))
        document.getElementById('askFollowup').addEventListener('click', runFollowup);
});

// =====================================================================
// NAVIGATION
// =====================================================================

function switchScreen(screenId) {
    // Only allow navigation to analysis pages if data is uploaded
    if (screenId !== 'homeScreen' && !uploadedData) {
        alert("Please upload an X-Ray image first.");
        return;
    }

    // Hide all screens
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));

    // Show target screen
    const target = document.getElementById(screenId);
    if (target) target.classList.add('active');

    // Update Nav Active State
    document.querySelectorAll('.nav-link').forEach(l => {
        l.classList.remove('active');
        if (l.getAttribute('onclick').includes(screenId)) {
            l.classList.add('active');
        }
    });

    // Auto-scroll to top
    window.scrollTo(0, 0);
}

function enableNavLinks() {
    document.querySelectorAll('.nav-link.disabled').forEach(link => {
        link.classList.remove('disabled');
    });
}

// =====================================================================
// UPLOAD
// =====================================================================

function initializeUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFileUpload(e.target.files[0]);
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    });
}

async function handleFileUpload(file) {
    try {
        const data = await API.uploadXray(file);
        uploadedData = data;


        // Show Success Message
        const successDiv = document.getElementById('uploadSuccess');
        successDiv.style.display = 'flex';

        // Enable Navigation
        enableNavLinks();

        // Populate base images where applicable (optional, usually done per page)
        // We defer rendering to each page's run function or pre-load if needed

    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed: ' + error.message);
    }
}

// =====================================================================
// FEATURE FUNCTIONS
// =====================================================================

async function runEnhancement() {
    if (!uploadedData) return;
    const method = document.getElementById('enhancementMethod').value;
    const data = await API.applyEnhancement(method);

    document.getElementById('enhancementBefore').src = data.images.original;
    document.getElementById('enhancementAfter').src = data.images.enhanced;
}

async function runNoise() {
    if (!uploadedData) {
        console.warn("runNoise called but uploadedData is null");
        return;
    }
    const method = document.getElementById('noiseMethod').value;
    console.log(`Starting runNoise with method: ${method}`);

    try {
        const data = await API.applyNoise(method);
        console.log("Noise API response received:", data);

        // 1. Update Metrics
        const metricsContainer = document.getElementById('noiseMetricsContainer');
        if (metricsContainer) {
            metricsContainer.style.display = 'block';
            if (data.metrics) {
                document.getElementById('metricResidual').textContent = data.metrics.residual_noise_level.toFixed(2);
                document.getElementById('metricVariance').textContent = data.metrics.variance_level.toFixed(2);
                document.getElementById('metricEdge').textContent = data.metrics.edge_preservation.toFixed(4);
            } else {
                console.warn("Metrics missing in response");
            }
        } else {
            console.error("noiseMetricsContainer not found in DOM");
        }

        // 2. Render Images & Maps
        if (uploadedData && uploadedData.images && uploadedData.images.original) {
            const originalImg = document.getElementById('noiseOriginal');
            if (originalImg) originalImg.src = uploadedData.images.original;
            else console.error("noiseOriginal element not found");
        }

        // Render Filtered (Grayscale)
        console.log("Rendering filtered image...");
        renderHeatmap(document.getElementById('noiseFilteredCanvas'), data.filtered_image, false);

        // Render Maps (Heatmaps)
        console.log("Rendering heatmaps...");

        // Logic for Residual Map switching
        let residualData = data.residual_map;
        let residualLabel = "Residual Noise (Normalized)";
        // Use mean_diff from response (or metric)
        const meanDiff = data.metrics.residual_noise_level;

        if (meanDiff < 2.0) {
            console.warn(`Low residual noise (${meanDiff.toFixed(2)}). Switching to Amplified view.`);
            if (data.residual_amplified) {
                residualData = data.residual_amplified;
                residualLabel = "Residual Noise (Amplified x4)";
            }
        }

        // Update Header
        const resHeader = document.getElementById('residualHeader');
        if (resHeader) resHeader.textContent = residualLabel;

        renderHeatmap(document.getElementById('residualMap'), residualData, true);
        renderHeatmap(document.getElementById('varianceMap'), data.local_variance_map, true);
        renderHeatmap(document.getElementById('gradientMap'), data.gradient_diff_map, true);
        console.log("Finished runNoise");

    } catch (err) {
        console.error("Noise Error:", err);
        alert("Failed to apply noise filter: " + err.message);
    }
}

/**
 * Render 2D array to canvas as Heatmap or Grayscale
 * @param {HTMLCanvasElement} canvas 
 * @param {Array} data - 2D array (list of lists)
 * @param {boolean} isHeatmap - true for Jet colormap, false for Grayscale
 */
function renderHeatmap(canvas, data, isHeatmap) {
    if (!canvas) {
        console.error("renderHeatmap: canvas is null");
        return;
    }
    if (!data) {
        console.error("renderHeatmap: data is null");
        return;
    }
    if (!data.length) {
        console.error("renderHeatmap: data is empty array");
        return;
    }

    console.log(`renderHeatmap: Processing ${data.length}x${data[0].length} array. isHeatmap=${isHeatmap}`);

    const ctx = canvas.getContext('2d');
    const height = data.length;
    const width = data[0].length;

    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    const buffer = imageData.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let value = data[y][x];
            // Ensure bounds
            value = Math.max(0, Math.min(255, value));

            const index = (y * width + x) * 4;
            let r, g, b;

            if (isHeatmap) {
                // "Hot" Palette (Black -> Red -> Yellow)
                // Much better for sparse noise maps
                const n = value / 255;

                if (n < 0.5) {
                    // Black to Red
                    // 0.0 -> (0,0,0)
                    // 0.5 -> (255,0,0)
                    r = Math.floor(2 * n * 255);
                    g = 0;
                    b = 0;
                } else {
                    // Red to Yellow
                    // 0.5 -> (255,0,0)
                    // 1.0 -> (255,255,0)
                    r = 255;
                    g = Math.floor(2 * (n - 0.5) * 255);
                    b = 0;
                }
            } else {
                // Grayscale
                r = g = b = value;
            }

            buffer[index] = r;
            buffer[index + 1] = g;
            buffer[index + 2] = b;
            buffer[index + 3] = 255; // Alpha
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

async function runSegmentation() {
    if (!uploadedData) return;
    const data = await API.getSegmentation();

    document.getElementById('segOtsu').src = data.images.otsu_mask;
    document.getElementById('segWatershed').src = data.images.watershed_mask;
    document.getElementById('segEdges').src = data.images.edges;

    if (data.watershed.region_stats) {
        Charts.createRegionChart('regionChart', data.watershed.region_stats);
    }
}

// =====================================================================
// ADVANCED HISTOGRAM ANALYSIS (FIXED)
// =====================================================================

// Stage configuration with EXACT backend paths
const HISTOGRAM_STAGES = [
    { label: 'CLAHE', key: 'enhancement.clahe', color: '#3bc9c9' },
    { label: 'Histogram Equalization', key: 'enhancement.hist_eq', color: '#66d9e8' },
    { label: 'Adaptive CLAHE', key: 'enhancement.adaptive_clahe', color: '#74c0fc' },
    { label: 'Gaussian Blur', key: 'noise_reduction.gaussian', color: '#51cf66' },
    { label: 'Median Filter', key: 'noise_reduction.median', color: '#94d82d' },
    { label: 'Bilateral Filter', key: 'noise_reduction.bilateral', color: '#ffd43b' },
    { label: 'OTSU Mask', key: 'segmentation.otsu.mask', color: '#ff6b6b' },
    { label: 'OTSU Edges', key: 'segmentation.otsu.edges', color: '#fa5252' },
    { label: 'Watershed Mask', key: 'segmentation.watershed.mask', color: '#ff8787' },
    { label: 'Original Intensity', key: 'intensity.original', color: '#1EE3CF' },
    { label: 'Enhanced Intensity', key: 'intensity.enhanced', color: '#0fbfbf' }
];

async function runHistograms() {
    console.log("Starting runHistograms...");
    if (!uploadedData) {
        alert('Please upload an X-ray image first');
        return;
    }

    try {
        console.log("Fetching advanced analysis...");
        // Fetch advanced data from backend
        const response = await API.getAdvancedAnalysis();

        if (!response) {
            console.error("No response from getAdvancedAnalysis");
            alert('Failed to load advanced analysis data');
            return;
        }

        // Store in global window object for access by utility functions
        window.advancedData = response;
        console.log('Advanced data loaded into window.advancedData', window.advancedData);
        console.log('Advanced data keys:', Object.keys(window.advancedData));

        // Populate stage selector dropdown
        populateStageSelector();

        // Render initial stage (CLAHE)
        const initialStage = HISTOGRAM_STAGES[0];
        console.log("Rendering initial stage:", initialStage);
        renderHistogramStage(initialStage.key, initialStage.label, initialStage.color);

    } catch (err) {
        console.error('Histogram error:', err);
        alert('Failed to compute histograms: ' + err.message);
    }
}

function populateStageSelector() {
    const selector = document.getElementById('histogramStageSelector');
    if (!selector) {
        console.error('Stage selector not found');
        return;
    }

    selector.innerHTML = '';

    // Add all stages
    HISTOGRAM_STAGES.forEach((stage, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = stage.label;
        selector.appendChild(option);
    });

    // Set default selection (CLAHE)
    selector.value = '0';

    // Add change listener
    selector.onchange = function () {
        const selectedIndex = parseInt(this.value);
        const stage = HISTOGRAM_STAGES[selectedIndex];
        console.log("Stage changed to:", stage.label);
        renderHistogramStage(stage.key, stage.label, stage.color);
    };
}

function renderHistogramStage(stageKey, stageLabel, stageColor) {
    console.log(`renderHistogramStage called for: ${stageLabel} (${stageKey})`);

    if (!window.advancedData) {
        console.error("window.advancedData is missing!");
        return;
    }

    // Check if we're dealing with pre-computed histogram data
    if (stageKey.startsWith('intensity.')) {
        console.log("Using pre-computed path");
        // Use pre-computed histogram from backend
        renderPrecomputedHistogram(stageKey, stageLabel, stageColor);
    } else {
        console.log("Using computed path");
        // Compute histogram from pixel array
        renderComputedHistogram(stageKey, stageLabel, stageColor);
    }
}

function renderPrecomputedHistogram(stageKey, stageLabel, stageColor) {
    const parts = stageKey.split('.');
    let data = window.advancedData;

    for (const part of parts) {
        data = data[part];
    }

    if (!data || !data.histogram || !data.cumulative) {
        showToast(`No histogram data available for ${stageLabel}`);
        return;
    }

    // Update chart titles
    updateChartTitles(stageLabel);

    // Render charts with pre-computed data
    Charts.createHistogramChart('histogramChart', data.histogram, stageLabel, stageColor);
    Charts.createCDFChart('cdfChart', data.cumulative, stageLabel, stageColor);
}

function renderComputedHistogram(stageKey, stageLabel, stageColor) {
    // Extract pixel array using utility function
    const pixelArray = HistogramUtils.getPixelArray(stageKey);

    if (!pixelArray || pixelArray.length === 0) {
        showToast(`No pixel data available for ${stageLabel}`);
        console.error('Failed to extract pixel array for:', stageKey);
        return;
    }

    console.log(`Extracted ${pixelArray.length} pixels for ${stageLabel}`);

    // Compute histogram and CDF on frontend
    const histogram = HistogramUtils.computeHistogram(pixelArray);
    const cdf = HistogramUtils.computeCDF(histogram);

    // Update chart titles
    updateChartTitles(stageLabel);

    // Render charts
    Charts.createHistogramChart('histogramChart', histogram, stageLabel, stageColor);
    Charts.createCDFChart('cdfChart', cdf, stageLabel, stageColor);
}

function updateChartTitles(stageLabel) {
    const histTitle = document.getElementById('histogramTitle');
    const cdfTitle = document.getElementById('cdfTitle');

    if (histTitle) histTitle.textContent = `Intensity Histogram — ${stageLabel}`;
    if (cdfTitle) cdfTitle.textContent = `CDF — ${stageLabel}`;
}

function showToast(message) {
    alert(message);
    console.warn(message);
}

async function runPipelines() {
    if (!uploadedData) return;
    const data = await API.comparePipelines();

    const div = document.getElementById('pipelineResults');
    div.style.display = 'block';

    // Best Result
    const best = data.best_pipeline;
    document.getElementById('bestPipelineName').textContent = best.name;
    document.getElementById('bestPipelineImage').src = best.output_image;

    // Update Best Metrics Grid in Hero Section
    const bestMetricsContainer = document.querySelector('.best-result .metrics-grid');
    if (bestMetricsContainer) {
        bestMetricsContainer.innerHTML = `
            <div class="m-item"><span>Composite Score</span><strong>${best.composite_score.toFixed(3)}</strong></div>
            <div class="m-item"><span>PSNR</span><strong>${best.psnr.toFixed(2)} dB</strong></div>
            <div class="m-item"><span>SSIM</span><strong>${best.ssim.toFixed(3)}</strong></div>
            <div class="m-item"><span>Contrast</span><strong>${best.contrast.toFixed(1)}</strong></div>
            <div class="m-item"><span>Noise Level</span><strong>${best.noise_level.toFixed(2)}</strong></div>
            <div class="m-item"><span>MSE</span><strong>${best.mse.toFixed(1)}</strong></div>
        `;
    }

    // Grid
    const grid = document.getElementById('pipelineGrid');
    grid.innerHTML = '';

    Object.values(data.all_pipelines).forEach(p => {
        const item = document.createElement('div');
        item.className = 'card';
        item.innerHTML = `
            <div class="image-wrapper mb-2" style="margin-bottom:1rem">
                <img src="${p.output_image}">
            </div>
            <strong>${p.name}</strong>
            
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:1rem; font-size:0.85rem">
                <div>Score: <span style="color:var(--primary-color)">${p.composite_score.toFixed(3)}</span></div>
                <div>PSNR: ${p.psnr.toFixed(1)}</div>
                <div>SSIM: ${p.ssim.toFixed(2)}</div>
                <div>Noise: ${p.noise_level.toFixed(1)}</div>
            </div>
        `;
        grid.appendChild(item);
    });
}

async function runRAG() {
    if (!uploadedData) return;
    const data = await API.analyzeWithRAG();

    // Sidebar Metrics
    const sem = data.semantic_features;
    const list = document.getElementById('metricSidebar');
    list.innerHTML = `
        <li><span>Entropy</span> <strong>${sem.entropy.toFixed(2)}</strong></li>
        <li><span>Contrast</span> <strong>${sem.gradient_variance.toFixed(2)}</strong></li>
        <li><span>Energy</span> <strong>${sem.frequency_energy.toFixed(0)}</strong></li>
    `;

    // Feature List
    const fList = document.getElementById('semanticFeatures');
    fList.innerHTML = `<span class="badge">Symmetry: ${sem.symmetry_difference.toFixed(2)}</span>`;

    // Parse Markdown for report
    document.getElementById('ragExplanation').innerHTML = marked.parse(data.rag_explanation);
}

async function runFollowup() {
    const q = document.getElementById('followupQuestion').value;
    if (!q) return;

    const data = await API.askFollowup(q);
    const ans = document.getElementById('followupAnswer');
    ans.style.display = 'block';
    // Parse Markdown for answer
    ans.innerHTML = `<strong>Answer:</strong><br>${marked.parse(data.answer)}`;
}
