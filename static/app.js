// Network Intrusion Detection System - Frontend JavaScript

let statsData = {
    totalPredictions: 0,
    attacksDetected: 0,
    normalTraffic: 0
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
    setupEventListeners();
    loadSystemStats();
});

// Setup Event Listeners
function setupEventListeners() {
    // Single Prediction Form
    const singleForm = document.getElementById('single-predict-form');
    if (singleForm) {
        singleForm.addEventListener('submit', function(e) {
            e.preventDefault();
            predictSingle();
        });
    }

    // Batch Prediction Button
    const batchBtn = document.getElementById('batch-predict-btn');
    if (batchBtn) {
        batchBtn.addEventListener('click', predictBatch);
    }
}

// Load Model Information
function loadModelInfo() {
    fetch('/api/model-info')
        .then(response => response.json())
        .then(data => {
            const statusEl = document.getElementById('model-status');
            const modelTypeEl = document.getElementById('model-type');
            const featureCountEl = document.getElementById('feature-count');
            const modelStatusEl = document.getElementById('model-load-status');

            if (data.loaded) {
                statusEl.textContent = 'Model Loaded';
                statusEl.className = 'badge bg-success';
                modelTypeEl.textContent = data.model_type;
                featureCountEl.textContent = data.feature_count;
                modelStatusEl.innerHTML = '<span class="status-online">✓ Ready</span>';
            } else {
                statusEl.textContent = 'Model Not Loaded';
                statusEl.className = 'badge bg-danger';
                modelStatusEl.innerHTML = '<span class="status-offline">✗ Not Loaded</span>';
            }
        })
        .catch(error => {
            console.error('Error loading model info:', error);
            showAlert('Error loading model information', 'danger');
        });
}

// Single Prediction
function predictSingle() {
    const form = document.getElementById('single-predict-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);

    // Show loading state
    const resultDiv = document.getElementById('single-result');
    resultDiv.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div></div>';

    fetch('/api/predict/single', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            const isAttack = result.is_attack;
            const confidence = (result.confidence * 100).toFixed(2);
            const resultClass = isAttack ? 'attack' : 'normal';
            const resultText = isAttack ? '⚠️ ATTACK DETECTED' : '✓ NORMAL TRAFFIC';

            resultDiv.innerHTML = `
                <div class="prediction-result ${resultClass}">
                    <div class="result-title">${resultText}</div>
                    <div class="confidence-score">${confidence}%</div>
                    <div class="progress mb-2">
                        <div class="progress-bar" style="width: ${confidence}%"></div>
                    </div>
                    <small>Confidence Score</small>
                </div>
            `;

            // Update stats
            statsData.totalPredictions++;
            if (isAttack) {
                statsData.attacksDetected++;
            } else {
                statsData.normalTraffic++;
            }
            updateStatsDisplay();

            showAlert(`Prediction: ${result.prediction} (${confidence}% confidence)`, 
                     isAttack ? 'danger' : 'success');
        } else {
            showAlert('Prediction failed: ' + result.error, 'danger');
            resultDiv.innerHTML = '<p class="text-danger">Prediction failed</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error making prediction: ' + error.message, 'danger');
        resultDiv.innerHTML = '<p class="text-danger">Error making prediction</p>';
    });
}

// Batch Prediction
function predictBatch() {
    const fileInput = document.getElementById('batch-file');
    const file = fileInput.files[0];

    if (!file) {
        showAlert('Please select a file', 'warning');
        return;
    }

    // Show loading state
    const resultsDiv = document.getElementById('batch-results');
    resultsDiv.innerHTML = '<div class="card"><div class="card-body text-center"><div class="spinner-border" role="status"></div></div></div>';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict/batch', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            const stats = result.statistics;
            resultsDiv.innerHTML = generateBatchResultsHTML(stats, result.results);

            // Update global stats
            statsData.totalPredictions += stats.total_samples;
            statsData.attacksDetected += stats.attack_count;
            statsData.normalTraffic += stats.normal_count;
            updateStatsDisplay();

            showAlert(`Processed ${stats.total_samples} samples: ${stats.attack_count} attacks detected`, 'success');
        } else {
            showAlert('Batch prediction failed: ' + result.error, 'danger');
            resultsDiv.innerHTML = '<div class="alert alert-danger">Prediction failed</div>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error processing batch: ' + error.message, 'danger');
        resultsDiv.innerHTML = '<div class="alert alert-danger">Error processing batch</div>';
    });
}

// Generate Batch Results HTML
function generateBatchResultsHTML(stats, results) {
    return `
        <div class="card">
            <div class="card-header">
                <h5>Batch Analysis Results</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <div class="card text-white bg-primary mb-3">
                            <div class="card-body">
                                <h6 class="card-title">Total Samples</h6>
                                <h3>${stats.total_samples}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-white bg-danger mb-3">
                            <div class="card-body">
                                <h6 class="card-title">Attacks Found</h6>
                                <h3>${stats.attack_count}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-white bg-success mb-3">
                            <div class="card-body">
                                <h6 class="card-title">Normal Samples</h6>
                                <h3>${stats.normal_count}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-white bg-warning mb-3">
                            <div class="card-body">
                                <h6 class="card-title">Avg Confidence</h6>
                                <h3>${(stats.avg_confidence * 100).toFixed(1)}%</h3>
                            </div>
                        </div>
                    </div>
                </div>

                <h6 class="mt-4 mb-3">Detection Rate: <strong>${((stats.attack_count / stats.total_samples) * 100).toFixed(2)}%</strong></h6>
                <div class="progress mb-4">
                    <div class="progress-bar bg-danger" style="width: ${(stats.attack_count / stats.total_samples * 100).toFixed(2)}%"></div>
                </div>

                <h6>Sample Results (First 100):</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>Is Attack</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${results.slice(0, 10).map(r => `
                                <tr>
                                    <td><span class="badge ${r.Is_Attack === 'True' ? 'bg-danger' : 'bg-success'}">${r.Prediction}</span></td>
                                    <td>${(parseFloat(r.Confidence) * 100).toFixed(2)}%</td>
                                    <td>${r.Is_Attack}</td>
                                    <td><small>${r.Timestamp}</small></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

// Load System Statistics
function loadSystemStats() {
    fetch('/api/statistics')
        .then(response => response.json())
        .then(data => {
            updateStatsDisplay();
        })
        .catch(error => console.error('Error loading stats:', error));
}

// Update Statistics Display
function updateStatsDisplay() {
    const detectionRate = statsData.totalPredictions > 0 
        ? ((statsData.attacksDetected / statsData.totalPredictions) * 100).toFixed(2) 
        : 0;

    document.getElementById('total-predictions').textContent = statsData.totalPredictions;
    document.getElementById('attacks-detected').textContent = statsData.attacksDetected;
    document.getElementById('normal-traffic').textContent = statsData.normalTraffic;
    document.getElementById('detection-rate').textContent = detectionRate + '%';
}

// Show Alert
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    const alertId = 'alert-' + Date.now();
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    alertsContainer.innerHTML += alertHTML;

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alertEl = document.getElementById(alertId);
        if (alertEl) {
            alertEl.remove();
        }
    }, 5000);
}

// Health Check (Optional)
function healthCheck() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            console.log('System Health:', data);
        })
        .catch(error => console.error('Health check failed:', error));
}

// Periodic health check
setInterval(healthCheck, 30000);
