// =====================================================================
// API HELPER - FRONTEND TO BACKEND COMMUNICATION
// =====================================================================

const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentImageFile = null;

// =====================================================================
// UTILITY FUNCTIONS
// =====================================================================

function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function showError(message) {
    alert(`Error: ${message}`);
    console.error(message);
}

async function handleResponse(response) {
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API request failed');
    }
    return await response.json();
}

// =====================================================================
// API FUNCTIONS
// =====================================================================

/**
 * Upload X-ray image and perform basic processing
 */
async function uploadXray(file) {
    try {
        showLoading();
        currentImageFile = file;

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload-xray`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Apply enhancement method
 */
async function applyEnhancement(method) {
    if (!currentImageFile) {
        showError('Please upload an image first');
        return null;
    }

    try {
        showLoading();

        const formData = new FormData();
        formData.append('file', currentImageFile);

        const response = await fetch(`${API_BASE_URL}/enhancement/${method}`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Apply noise reduction filter
 */
async function applyNoise(method) {
    if (!currentImageFile) {
        showError('Please upload an image first');
        return null;
    }

    try {
        showLoading();

        const formData = new FormData();
        formData.append('file', currentImageFile);

        const response = await fetch(`${API_BASE_URL}/noise/${method}`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Get segmentation analysis
 */
async function getSegmentation() {
    if (!currentImageFile) {
        showError('Please upload an image first');
        return null;
    }

    try {
        showLoading();

        const formData = new FormData();
        formData.append('file', currentImageFile);

        const response = await fetch(`${API_BASE_URL}/segmentation`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Get histogram analysis
 */


/**
 * Run complete pipeline comparison
 */
async function comparePipelines() {
    if (!currentImageFile) {
        showError('Please upload an image first');
        return null;
    }

    try {
        showLoading();

        const formData = new FormData();
        formData.append('file', currentImageFile);

        const response = await fetch(`${API_BASE_URL}/pipelines`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Analyze with RAG
 */
async function analyzeWithRAG() {
    if (!currentImageFile) {
        showError('Please upload an image first');
        return null;
    }

    try {
        showLoading();

        const formData = new FormData();
        formData.append('file', currentImageFile);

        const response = await fetch(`${API_BASE_URL}/rag/analyze`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Ask follow-up question
 */
async function askFollowup(question) {
    try {
        showLoading();

        const formData = new FormData();
        formData.append('question', question);

        const response = await fetch(`${API_BASE_URL}/rag/followup`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Get advanced analysis with all enhancement/noise/segmentation stages
 */
async function getAdvancedAnalysis() {
    if (!currentImageFile) {
        showError('Please upload an image first');
        return null;
    }

    try {
        showLoading();

        const formData = new FormData();
        formData.append('file', currentImageFile);

        const response = await fetch(`${API_BASE_URL}/advanced`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result.data;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

/**
 * Export report
 */
async function exportReport(format, data) {
    try {
        showLoading();

        const formData = new FormData();
        formData.append('format', format);
        formData.append('data', JSON.stringify(data));

        const response = await fetch(`${API_BASE_URL}/export/report`, {
            method: 'POST',
            body: formData
        });

        const result = await handleResponse(response);
        hideLoading();

        return result;
    } catch (error) {
        hideLoading();
        showError(error.message);
        throw error;
    }
}

// =====================================================================
// EXPORT FUNCTIONS
// =====================================================================

window.API = {
    uploadXray,
    applyEnhancement,
    applyNoise,
    getSegmentation,
    comparePipelines,
    analyzeWithRAG,
    askFollowup,
    getAdvancedAnalysis,
    exportReport
};
