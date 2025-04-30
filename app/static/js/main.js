// Main JavaScript file for Churn Prediction App

document.addEventListener('DOMContentLoaded', function() {
    // Enable tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Enable popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Platform selection in prediction form
    const platformRadios = document.querySelectorAll('input[type=radio][name^="platform_"]');
    if (platformRadios.length > 0) {
        platformRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                // Reset all platform values to 0
                platformRadios.forEach(r => {
                    r.value = "0";
                });
                // Set selected platform value to 1
                this.value = "1";
            });
        });
    }
    
    // Form validation
    const predictionForm = document.querySelector('form[action="/predict"]');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            if (!predictionForm.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            predictionForm.classList.add('was-validated');
        }, false);
    }
});
