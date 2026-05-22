document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const demoBtn = document.getElementById('demo-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const uploadSection = document.getElementById('upload-section');
    const dashboard = document.getElementById('dashboard');
    const loader = document.getElementById('loader');
    
    const savingsVal = document.getElementById('savings-val');
    const spaceVal = document.getElementById('space-val');
    
    let chartInstance = null;

    // Drag and drop handlers
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    demoBtn.addEventListener('click', () => {
        runPrediction(null);
    });

    resetBtn.addEventListener('click', () => {
        dashboard.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        if (chartInstance) chartInstance.destroy();
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a CSV file.');
            return;
        }
        runPrediction(file);
    }

    async function runPrediction(file) {
        uploadSection.classList.add('hidden');
        loader.classList.remove('hidden');
        
        try {
            const formData = new FormData();
            if (file) {
                formData.append('file', file);
            }
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: file ? formData : null
            });
            
            if (!response.ok) throw new Error('API Error');
            const result = await response.json();
            
            if (result.status === 'success') {
                renderDashboard(result);
            }
        } catch (error) {
            alert('Error running predictions: ' + error.message);
            loader.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    }

    function renderDashboard(result) {
        loader.classList.add('hidden');
        dashboard.classList.remove('hidden');
        
        // Animate numbers
        animateValue(savingsVal, 0, result.estimated_savings, 1500, '$');
        animateValue(spaceVal, 0, result.estimated_savings * 0.12, 1500, '', ' sq ft');
        
        renderChart(result.historical, result.data);
    }

    function animateValue(obj, start, end, duration, prefix = '', suffix = '') {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = Math.floor(progress * (end - start) + start);
            obj.innerHTML = prefix + current.toLocaleString() + suffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    function renderChart(historical, predictions) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        if (chartInstance) chartInstance.destroy();
        
        const recentHistorical = historical.slice(-20);
        const labels = [];
        const actualData = [];
        const predictedData = [];
        
        recentHistorical.forEach(item => {
            labels.push('Day ' + item.day);
            actualData.push(item.actual);
            predictedData.push(null);
        });
        
        if (recentHistorical.length > 0) {
            predictedData[predictedData.length - 1] = recentHistorical[recentHistorical.length - 1].actual;
        }

        predictions.forEach(item => {
            labels.push('Day ' + item.day);
            actualData.push(item.actual); 
            predictedData.push(item.final_predicted);
        });

        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Actual Sales',
                        data: actualData,
                        borderColor: '#94a3b8',
                        backgroundColor: 'rgba(148, 163, 184, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'LSTM Dual-Model Prediction',
                        data: predictedData,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#ffffff',
                        pointBorderColor: '#3b82f6',
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleFont: { size: 14 },
                        bodyFont: { size: 13 },
                        padding: 12,
                        cornerRadius: 8
                    }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    x: { grid: { display: false } }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
});
