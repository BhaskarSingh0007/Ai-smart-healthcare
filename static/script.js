document.querySelector("form").addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevents the page from refreshing/reloading

    // 1. Automatically grab all form data using the 'name' attributes in your HTML
    const formData = new FormData(e.target);
    
    // 2. Convert form data into a clean JSON object
    const data = {};
    formData.forEach((value, key) => {
        /** * We convert all inputs to numbers (parseFloat). 
         * This is vital because machine learning models cannot process "strings".
         */
        data[key] = parseFloat(value);
    });

    // Log the data to the console so you can check it during your demo (F12)
    console.log("Sending data to server:", data);

    try {
        // 3. Send the data to your Flask Server (/predict route in main.py)
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Server Error");
        }

        const result = await response.json();

        // 4. Display the result dynamically on the page
        const resultArea = document.getElementById("result");
        
        if (resultArea) {
            resultArea.style.display = "block";
            
            // Set color: Red for disease, Green for healthy
            const bgColor = result.prediction === 'Heart Disease Detected' ? '#ffebee' : '#e8f5e9';
            const borderColor = result.prediction === 'Heart Disease Detected' ? '#f44336' : '#4caf50';

            resultArea.innerHTML = `
                <div style="padding: 20px; border-radius: 8px; background-color: ${bgColor}; border: 2px solid ${borderColor}; margin-top: 20px;">
                    <h3 style="color: #333; margin-top: 0;">Diagnosis Result</h3>
                    <p style="font-size: 1.1em;"><b>Status:</b> ${result.prediction}</p>
                    <p style="font-size: 1.1em;"><b>Confidence:</b> ${(result.probability * 100).toFixed(2)}%</p>
                </div>
            `;
            
            // Auto-scroll to the result so the user sees it
            resultArea.scrollIntoView({ behavior: 'smooth' });

        } else {
            // Fallback alert if the 'result' div is missing from HTML
            alert(`Prediction: ${result.prediction}\nConfidence: ${(result.probability * 100).toFixed(2)}%`);
        }

    } catch (error) {
        console.error("Prediction Error:", error);
        alert("Error: " + error.message + ". Please ensure all fields are filled and the server is running.");
    }
});