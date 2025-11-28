// DOM Elements
const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const enhanceBtn = document.getElementById('enhanceBtn');
const downloadBtn = document.getElementById('downloadBtn');
const imagePreview = document.getElementById('imagePreview');
const originalImage = document.getElementById('originalImage');
const enhancedImage = document.getElementById('enhancedImage');
const imagePlaceholder = document.getElementById('imagePlaceholder');
const loadingOverlay = document.getElementById('loadingOverlay');

// State
let selectedFile = null;
let originalImageUrl = null;
let enhancedImageUrl = null;

// Event Listeners
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

uploadSection.addEventListener('click', (e) => {
    if (e.target !== browseBtn && !browseBtn.contains(e.target)) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and Drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// File handling
function handleFile(file) {
    // Validate file type
    if (!file.type.match(/^image\/(png|jpeg|jpg)$/i)) {
        alert('Please select a PNG or JPG image file.');
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        originalImageUrl = e.target.result;
        originalImage.src = originalImageUrl;
        enhancedImage.src = '';
        enhancedImage.classList.remove('loaded');
        enhancedImage.style.display = 'none';
        imagePlaceholder.classList.remove('hidden');
        imagePreview.style.display = 'block';
        enhanceBtn.disabled = false;
        downloadBtn.disabled = true;
        enhancedImageUrl = null;
    };

    reader.readAsDataURL(file);
}

// Enhance button
enhanceBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    // Show loading
    loadingOverlay.style.display = 'flex';
    enhanceBtn.disabled = true;
    downloadBtn.disabled = true;

    try {
        const response = await fetch('/enhance', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to enhance image');
        }

        // Get the enhanced image as blob
        const blob = await response.blob();
        enhancedImageUrl = URL.createObjectURL(blob);
        
        // Show both images side by side
        originalImage.src = originalImageUrl;
        enhancedImage.src = enhancedImageUrl;
        enhancedImage.onload = () => {
            imagePlaceholder.classList.add('hidden');
            enhancedImage.style.display = 'block';
            enhancedImage.classList.add('loaded');
        };
        
        // Enable download button
        downloadBtn.disabled = false;
        
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Enhancement error:', error);
    } finally {
        loadingOverlay.style.display = 'none';
        enhanceBtn.disabled = false;
    }
});

// Download button
downloadBtn.addEventListener('click', () => {
    if (!enhancedImageUrl) {
        alert('No enhanced image available. Please enhance the image first.');
        return;
    }

    const downloadLink = document.createElement('a');
    downloadLink.href = enhancedImageUrl;
    downloadLink.download = 'enhanced_image.png';
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
});

