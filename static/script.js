const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const maskDropZone = document.getElementById('mask-drop-zone');
const maskFileInput = document.getElementById('mask-file-input');

const resultsSection = document.getElementById('results');
const originalImg = document.getElementById('original-image');
const maskOverlay = document.getElementById('mask-overlay');
const realtimeMetrics = document.getElementById('realtime-metrics');
const resetBtn = document.getElementById('reset-btn');
const runBtn = document.getElementById('run-btn');
const modelIndicator = document.getElementById('model-indicator');

let currentFile = null;
let currentMaskFile = null;

// Handle Primary Image Upload
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) stageFile(e.dataTransfer.files[0], 'image');
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) stageFile(e.target.files[0], 'image');
});

// Handle Mask Upload
maskDropZone.addEventListener('click', () => maskFileInput.click());
maskDropZone.addEventListener('dragover', (e) => { e.preventDefault(); maskDropZone.classList.add('dragover'); });
maskDropZone.addEventListener('dragleave', () => maskDropZone.classList.remove('dragover'));
maskDropZone.addEventListener('drop', (e) => {
    e.preventDefault(); maskDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) stageFile(e.dataTransfer.files[0], 'mask');
});
maskFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) stageFile(e.target.files[0], 'mask');
});

function stageFile(file, type) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    
    if (type === 'image') {
        currentFile = file;
        dropZone.classList.add('has-file');
        dropZone.querySelector('p').innerHTML = `<strong>${file.name}</strong><br>Ready for inference`;
        runBtn.disabled = false;
        runBtn.classList.remove('disabled');
        modelIndicator.innerHTML = `<span class="dot"></span> Image loaded successfully`;
        
        // Pre-load image for results
        const reader = new FileReader();
        reader.onload = (e) => { originalImg.src = e.target.result; };
        reader.readAsDataURL(file);
    } else if (type === 'mask') {
        currentMaskFile = file;
        maskDropZone.classList.add('has-file');
        maskDropZone.querySelector('p').innerHTML = `<strong>${file.name}</strong><br>Mask loaded for evaluation`;
    }
}

// Run Prediction
runBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    runBtn.disabled = true;
    runBtn.innerHTML = `Running...`;
    modelIndicator.innerHTML = `<span class="dot"></span> Processing inference on backend...`;
    
    const formData = new FormData();
    formData.append('file', currentFile);
    if (currentMaskFile) {
        formData.append('mask_file', currentMaskFile);
    }

    try {
        // Use the absolute URL so it works even if the HTML is opened directly from the file system
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        // Display Results Overlay
        maskOverlay.src = `data:image/png;base64,${data.mask}`;
        maskOverlay.style.opacity = 1;
        
        // Display Raw Masks Comparison
        const masksComparison = document.getElementById('masks-comparison');
        masksComparison.classList.remove('hidden');
        document.getElementById('predicted-raw-mask').src = `data:image/png;base64,${data.raw_mask}`;
        
        // Display Metrics and GT Mask if available
        const gtMaskCol = document.getElementById('gt-mask-col');
        if (data.metrics) {
            document.getElementById('rt-dice').innerText = (data.metrics.dice * 100).toFixed(1) + '%';
            document.getElementById('rt-iou').innerText = (data.metrics.iou * 100).toFixed(1) + '%';
            document.getElementById('rt-precision').innerText = (data.metrics.precision * 100).toFixed(1) + '%';
            document.getElementById('rt-recall').innerText = (data.metrics.recall * 100).toFixed(1) + '%';
            realtimeMetrics.classList.remove('hidden');
            
            if (data.gt_mask) {
                document.getElementById('gt-raw-mask').src = `data:image/png;base64,${data.gt_mask}`;
                gtMaskCol.style.display = 'flex';
            } else {
                gtMaskCol.style.display = 'none';
            }
        } else {
            realtimeMetrics.classList.add('hidden');
            gtMaskCol.style.display = 'none';
        }
        
        // Hide upload panel, show results
        document.querySelector('.upload-panel').classList.add('hidden');
        resultsSection.classList.remove('hidden');

    } catch (error) {
        console.error(error);
        alert('Failed to analyze the image: ' + error.message);
        runBtn.disabled = false;
        runBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Prediction`;
        modelIndicator.innerHTML = `<span class="dot"></span> Image loaded successfully`;
    }
});

// Handle Reset
resetBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    document.querySelector('.upload-panel').classList.remove('hidden');
    
    fileInput.value = '';
    maskFileInput.value = '';
    currentFile = null;
    currentMaskFile = null;
    
    dropZone.classList.remove('has-file');
    dropZone.querySelector('p').innerHTML = `<strong>Drag & drop aerial imagery</strong><br>or click to browse`;
    
    maskDropZone.classList.remove('has-file');
    maskDropZone.querySelector('p').innerHTML = `<strong>Drag & drop mask file</strong><br>For comparison metrics`;
    
    runBtn.disabled = true;
    runBtn.classList.add('disabled');
    runBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Prediction`;
    
    modelIndicator.innerHTML = `<span class="dot grey"></span> Waiting for input...`;
    maskOverlay.style.opacity = 0;
    document.getElementById('masks-comparison').classList.add('hidden');
});
