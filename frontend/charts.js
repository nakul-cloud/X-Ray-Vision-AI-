// =====================================================================
// CHARTS HELPER - CHART.JS VISUALIZATIONS
// =====================================================================

// Chart instances storage
const chartInstances = {};

// Color palette
const colors = {
    primary: '#1EE3CF',
    secondary: '#67FFF1',
    muted: '#A4C7C7',
    background: 'rgba(30, 227, 207, 0.1)',
    border: 'rgba(30, 227, 207, 0.5)'
};

// Default chart options
const defaultOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
        legend: {
            labels: {
                color: colors.muted,
                font: {
                    family: 'Inter, sans-serif',
                    size: 12
                }
            }
        }
    },
    scales: {
        x: {
            ticks: { color: colors.muted },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
            ticks: { color: colors.muted },
            grid: { color: 'rgba(255, 255, 255, 0.1)' }
        }
    }
};

// =====================================================================
// CHART CREATION FUNCTIONS
// =====================================================================

/**
 * Create histogram chart for single stage
 */
function createHistogramChart(canvasId, data, label, color = colors.primary) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 256 }, (_, i) => i),
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: color + '33',  // 20% opacity
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...defaultOptions,
            plugins: {
                ...defaultOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label} - Intensity: ${context.label}, Frequency: ${context.raw}`;
                        }
                    }
                }
            },
            scales: {
                ...defaultOptions.scales,
                x: {
                    ...defaultOptions.scales.x,
                    title: { display: true, text: 'Intensity', color: colors.muted }
                },
                y: {
                    ...defaultOptions.scales.y,
                    title: { display: true, text: 'Frequency', color: colors.muted }
                }
            }
        }
    });
}

/**
 * Create CDF chart for single stage
 */
function createCDFChart(canvasId, data, label, color = colors.secondary) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 256 }, (_, i) => i),
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: color + '1A',  // 10% opacity
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...defaultOptions,
            plugins: {
                ...defaultOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label} - Intensity: ${context.label}, Cumulative: ${context.raw.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                ...defaultOptions.scales,
                x: {
                    ...defaultOptions.scales.x,
                    title: { display: true, text: 'Intensity', color: colors.muted }
                },
                y: {
                    ...defaultOptions.scales.y,
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Cumulative %',
                        color: colors.muted
                    }
                }
            }
        }
    });
}

// Helper to convert hex to rgb for rgba transparency
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ?
        `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` :
        '30, 227, 207';
}

/**
 * Create region area bar chart
 */
function createRegionChart(canvasId, regionStats) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    const labels = regionStats.map(r => `Region ${r.label}`);
    const areas = regionStats.map(r => r.area);

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Area (pixels)',
                data: areas,
                backgroundColor: colors.background,
                borderColor: colors.primary,
                borderWidth: 2
            }]
        },
        options: {
            ...defaultOptions,
            indexAxis: 'y'
        }
    });
}

/**
 * Create contrast comparison bar chart
 */
function createContrastChart(canvasId, pipelines, rankings) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    // Sort by rankings
    const sortedData = rankings.slice(0, 9).map(([pid, _]) => {
        const pipeline = pipelines[pid];
        return {
            name: pipeline.name,
            contrast: pipeline.contrast
        };
    });

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedData.map(d => d.name),
            datasets: [{
                label: 'Contrast',
                data: sortedData.map(d => d.contrast),
                backgroundColor: colors.background,
                borderColor: colors.primary,
                borderWidth: 2
            }]
        },
        options: {
            ...defaultOptions,
            indexAxis: 'y',
            plugins: {
                ...defaultOptions.plugins,
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Create PSNR vs SSIM scatter chart
 */
function createPSNRSSIMChart(canvasId, pipelines, rankings) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    const data = Object.values(pipelines).map(p => ({
        x: p.ssim,
        y: p.psnr,
        r: p.composite_score * 20 // Scale for bubble size
    }));

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Pipelines',
                data: data,
                backgroundColor: colors.background,
                borderColor: colors.primary,
                borderWidth: 2
            }]
        },
        options: {
            ...defaultOptions,
            scales: {
                x: {
                    ...defaultOptions.scales.x,
                    title: {
                        display: true,
                        text: 'SSIM',
                        color: colors.muted
                    }
                },
                y: {
                    ...defaultOptions.scales.y,
                    title: {
                        display: true,
                        text: 'PSNR (dB)',
                        color: colors.muted
                    }
                }
            }
        }
    });
}

/**
 * Create composite score bar chart
 */
function createCompositeChart(canvasId, pipelines, rankings) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    const sortedData = rankings.map(([pid, score]) => ({
        name: pipelines[pid].name,
        score: score
    }));

    // Highlight best pipeline
    const backgroundColors = sortedData.map((_, i) =>
        i === 0 ? colors.primary : colors.background
    );

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedData.map(d => d.name),
            datasets: [{
                label: 'Composite Score',
                data: sortedData.map(d => d.score),
                backgroundColor: backgroundColors,
                borderColor: colors.primary,
                borderWidth: 2
            }]
        },
        options: {
            ...defaultOptions,
            indexAxis: 'y',
            plugins: {
                ...defaultOptions.plugins,
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Create normalized metrics grouped bar chart
 */
function createNormalizedChart(canvasId, pipelines, rankings) {
    destroyChart(canvasId);

    const ctx = document.getElementById(canvasId).getContext('2d');

    // Get top 5 pipelines
    const topPipelines = rankings.slice(0, 5).map(([pid, _]) => pipelines[pid]);

    // Normalize metrics to 0-1 scale
    const maxContrast = Math.max(...topPipelines.map(p => p.contrast));
    const maxPSNR = Math.max(...topPipelines.map(p => p.psnr));

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topPipelines.map(p => p.name),
            datasets: [
                {
                    label: 'Contrast (norm)',
                    data: topPipelines.map(p => p.contrast / maxContrast),
                    backgroundColor: 'rgba(30, 227, 207, 0.6)',
                    borderColor: colors.primary,
                    borderWidth: 1
                },
                {
                    label: 'PSNR (norm)',
                    data: topPipelines.map(p => p.psnr / maxPSNR),
                    backgroundColor: 'rgba(103, 255, 241, 0.6)',
                    borderColor: colors.secondary,
                    borderWidth: 1
                },
                {
                    label: 'SSIM',
                    data: topPipelines.map(p => p.ssim),
                    backgroundColor: 'rgba(164, 199, 199, 0.6)',
                    borderColor: colors.muted,
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...defaultOptions,
            scales: {
                ...defaultOptions.scales,
                y: {
                    ...defaultOptions.scales.y,
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Normalized Value',
                        color: colors.muted
                    }
                }
            }
        }
    });
}

/**
 * Destroy existing chart instance
 */
function destroyChart(canvasId) {
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
        delete chartInstances[canvasId];
    }
}

/**
 * Destroy all charts
 */
function destroyAllCharts() {
    Object.keys(chartInstances).forEach(id => {
        chartInstances[id].destroy();
    });
    chartInstances = {};
}

// =====================================================================
// EXPORT FUNCTIONS
// =====================================================================

window.Charts = {
    createHistogramChart,
    createCDFChart,
    createRegionChart,
    createContrastChart,
    createPSNRSSIMChart,
    createCompositeChart,
    createNormalizedChart,
    destroyChart,
    destroyAllCharts
};
