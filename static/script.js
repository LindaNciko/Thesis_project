<<<<<<< HEAD
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultsArea = document.getElementById('results-area');
    const resultsContent = document.getElementById('results-content');
    const errorArea = document.getElementById('error-area');
    const errorContent = document.getElementById('error-content');
    const submitButton = document.getElementById('submit-button');
    const outputPlaceholder = document.getElementById('output-placeholder'); // Get placeholder

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default page reload

        // Clear previous results and errors, hide areas, show placeholder
        resultsArea.style.display = 'none';
        errorArea.style.display = 'none';
        outputPlaceholder.style.display = 'flex'; // Show placeholder initially
        resultsContent.textContent = '';
        errorContent.textContent = '';
        submitButton.disabled = true; // Disable button during request
        submitButton.textContent = 'Predicting...';

        // Gather form data
        const selects = form.querySelectorAll('select');
        const payload = {};
        selects.forEach(select => {
            const key = select.getAttribute('data-alias') || select.name;
            payload[key] = select.value;
        });

        console.log('Sending payload:', payload);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            outputPlaceholder.style.display = 'none'; // Hide placeholder after fetch attempt

            if (response.ok) {
                const data = await response.json();
                console.log('Received prediction:', data);
                resultsContent.textContent = JSON.stringify(data, null, 2);
                resultsArea.style.display = 'block'; // Show results area
            } else {
                console.error('Prediction request failed with status:', response.status);
                let errorData;
                try {
                     errorData = await response.json();
                     console.error('Error details:', errorData);
                     errorContent.textContent = errorData.detail || JSON.stringify(errorData);
                     if (errorData.detail && typeof errorData.detail === 'object' && errorData.detail.validation_errors) {
                         errorContent.textContent = `Validation Error: ${JSON.stringify(errorData.detail.validation_errors)}`;
                     } else if (typeof errorData.detail === 'string') {
                          errorContent.textContent = errorData.detail;
                     }
                } catch (e) {
                     errorContent.textContent = `Server responded with status ${response.status}. ${response.statusText || 'No further details.'}`;
                }
                errorArea.style.display = 'block'; // Show error area
            }
        } catch (error) {
             outputPlaceholder.style.display = 'none'; // Hide placeholder on error too
            console.error('Network or fetch error:', error);
            errorContent.textContent = `An error occurred: ${error.message}. Check the console and ensure the backend is running.`;
            errorArea.style.display = 'block'; // Show error area
        } finally {
             submitButton.disabled = false;
             submitButton.textContent = 'Predict';
        }
    });
=======
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultsArea = document.getElementById('results-area');
    const resultsContent = document.getElementById('results-content');
    const errorArea = document.getElementById('error-area');
    const errorContent = document.getElementById('error-content');
    const submitButton = document.getElementById('submit-button');
    const outputPlaceholder = document.getElementById('output-placeholder'); // Get placeholder

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default page reload

        // Clear previous results and errors, hide areas, show placeholder
        resultsArea.style.display = 'none';
        errorArea.style.display = 'none';
        outputPlaceholder.style.display = 'flex'; // Show placeholder initially
        resultsContent.textContent = '';
        errorContent.textContent = '';
        submitButton.disabled = true; // Disable button during request
        submitButton.textContent = 'Predicting...';

        // Gather form data
        const selects = form.querySelectorAll('select');
        const payload = {};
        selects.forEach(select => {
            const key = select.getAttribute('data-alias') || select.name;
            payload[key] = select.value;
        });

        console.log('Sending payload:', payload);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            outputPlaceholder.style.display = 'none'; // Hide placeholder after fetch attempt

            if (response.ok) {
                const data = await response.json();
                console.log('Received prediction:', data);
                resultsContent.textContent = JSON.stringify(data, null, 2);
                resultsArea.style.display = 'block'; // Show results area
            } else {
                console.error('Prediction request failed with status:', response.status);
                let errorData;
                try {
                     errorData = await response.json();
                     console.error('Error details:', errorData);
                     errorContent.textContent = errorData.detail || JSON.stringify(errorData);
                     if (errorData.detail && typeof errorData.detail === 'object' && errorData.detail.validation_errors) {
                         errorContent.textContent = `Validation Error: ${JSON.stringify(errorData.detail.validation_errors)}`;
                     } else if (typeof errorData.detail === 'string') {
                          errorContent.textContent = errorData.detail;
                     }
                } catch (e) {
                     errorContent.textContent = `Server responded with status ${response.status}. ${response.statusText || 'No further details.'}`;
                }
                errorArea.style.display = 'block'; // Show error area
            }
        } catch (error) {
             outputPlaceholder.style.display = 'none'; // Hide placeholder on error too
            console.error('Network or fetch error:', error);
            errorContent.textContent = `An error occurred: ${error.message}. Check the console and ensure the backend is running.`;
            errorArea.style.display = 'block'; // Show error area
        } finally {
             submitButton.disabled = false;
             submitButton.textContent = 'Predict';
        }
    });
>>>>>>> d458a7e9b212d2945525c7e3f6e22f03e5c5254d
});