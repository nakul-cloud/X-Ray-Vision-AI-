// =====================================================================
// HISTOGRAM UTILITIES - FRONTEND COMPUTATION (FIXED)
// =====================================================================

/**
 * Extract pixel array from nested object using dot notation key
 * @param {string} stageKey - Dot notation path (e.g., "enhancement.clahe")
 * @returns {Array|null} - Flattened pixel array or null if not found
 */
function getPixelArray(stageKey) {
    if (!window.advancedData) {
        console.error('No advanced data available');
        return null;
    }

    const parts = stageKey.split(".");
    let obj = window.advancedData;

    // Navigate through nested object
    for (const part of parts) {
        if (obj && typeof obj === 'object' && part in obj) {
            obj = obj[part];
        } else {
            console.error(`Path not found: ${stageKey} (failed at: ${part})`);
            return null;
        }
    }

    // Handle different data types
    if (!obj) {
        return null;
    }

    // If it's already a histogram object with pre-computed data
    if (obj.histogram && Array.isArray(obj.histogram)) {
        return obj.histogram;
    }

    // If it's a 2D pixel array, flatten it
    if (Array.isArray(obj)) {
        if (Array.isArray(obj[0])) {
            // 2D array - flatten it
            return obj.flat();
        } else {
            // Already 1D array
            return obj;
        }
    }

    console.error(`Invalid data type for ${stageKey}:`, typeof obj);
    return null;
}

/**
 * Compute intensity histogram from pixel array
 * @param {Array} pixelArray - 1D array of pixel values (0-255)
 * @returns {Array} - 256-bin histogram
 */
function computeHistogram(pixelArray) {
    if (!pixelArray || !Array.isArray(pixelArray) || pixelArray.length === 0) {
        console.error('Invalid pixel array for histogram computation');
        return new Array(256).fill(0);
    }

    const histogram = new Array(256).fill(0);

    for (let i = 0; i < pixelArray.length; i++) {
        const value = Math.floor(pixelArray[i]);
        if (value >= 0 && value < 256) {
            histogram[value]++;
        }
    }

    return histogram;
}

/**
 * Compute CDF from histogram
 * @param {Array} histogram - 256-bin histogram
 * @returns {Array} - CDF normalized to 0-100%
 */
function computeCDF(histogram) {
    if (!histogram || !Array.isArray(histogram) || histogram.length !== 256) {
        console.error('Invalid histogram for CDF computation');
        return new Array(256).fill(0);
    }

    const total = histogram.reduce((sum, val) => sum + val, 0);
    if (total === 0) {
        return new Array(256).fill(0);
    }

    const cdf = [];
    let cumulative = 0;

    for (let i = 0; i < 256; i++) {
        cumulative += histogram[i];
        cdf.push((cumulative / total) * 100);
    }

    return cdf;
}

// Export functions
window.HistogramUtils = {
    getPixelArray,
    computeHistogram,
    computeCDF
};
