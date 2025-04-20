document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('#uploadForm');
    const errorDiv = document.querySelector('.error');
    const button = document.querySelector('button');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.querySelector('input[type="file"]');
            const ageInput = document.querySelector('input[type="number"]');
            const genderInput = document.querySelector('select');
            const loadingIcon = button.querySelector('i');

            errorDiv.style.display = 'none';
            errorDiv.textContent = '';

            if (!fileInput.files.length) {
                errorDiv.textContent = 'Please upload an X-ray image.';
                errorDiv.style.display = 'block';
                return;
            }
            if (!ageInput.value || ageInput.value < 0 || ageInput.value > 120) {
                errorDiv.textContent = 'Please enter a valid age (0-120).';
                errorDiv.style.display = 'block';
                return;
            }
            if (!genderInput.value) {
                errorDiv.textContent = 'Please select a gender.';
                errorDiv.style.display = 'block';
                return;
            }

            button.disabled = true;
            loadingIcon.style.display = 'inline-block';
            button.textContent = 'Predicting';

            const formData = new FormData(form);
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.style.display = 'block';
                } else {
                    window.location.href = `/dashboard?prediction=${data.prediction}&confidence=${data.confidence}&heatmap=${encodeURIComponent(data.heatmap || '')}`;
                }
            } catch (error) {
                errorDiv.textContent = 'An error occurred. Please try again.';
                errorDiv.style.display = 'block';
            } finally {
                button.disabled = false;
                loadingIcon.style.display = 'none';
                button.textContent = 'Predict';
            }
        });
    }

    if (window.location.pathname === '/dashboard') {
        const urlParams = new URLSearchParams(window.location.search);
        document.getElementById('prediction').textContent = urlParams.get('prediction') || '{{ prediction }}';
        document.getElementById('confidence').textContent = urlParams.get('confidence') || '{{ confidence }}';
        document.getElementById('heatmap').src = decodeURIComponent(urlParams.get('heatmap') || '{{ heatmap }}');
    }
});