
document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predictBtn');
    const refreshBtn = document.getElementById('refreshBtn'); 
    const resultBox = document.getElementById('resultBox');
    const summaryText = document.getElementById('summaryText');
    const chartCanvas = document.getElementById('probChart');
    let probabilityChart = null; 


    function updateChart(labels, data) {
        if (probabilityChart) {
            probabilityChart.destroy();
        }

        probabilityChart = new Chart(chartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: data.map(p => p * 100),
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Probability (%)' },
                        max: 100
                    },
                    x: {
                        title: { display: true, text: 'Predicted Score Range' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Score Distribution Likelihood' }
                }
            }
        });
    }

    async function handlePrediction() {
        const batTeam = document.getElementById('bat_team').value;
        const bowlTeam = document.getElementById('bowl_team').value;
        const venue = document.getElementById('venue').value;
        
        const overs = parseFloat(document.getElementById('overs').value);
        const runsLast5 = parseInt(document.getElementById('runs_last_5').value, 10);
        const wicketsLast5 = parseInt(document.getElementById('wickets_last_5').value, 10);

        if (batTeam === bowlTeam) {
            alert("Batting Team and Bowling Team must be different!");
            return;
        }
        
        if (isNaN(overs) || isNaN(runsLast5) || isNaN(wicketsLast5) || 
            overs < 5 || overs > 20 || runsLast5 < 0 || wicketsLast5 < 0) {
            alert("Please ensure all fields contain valid numbers. Overs must be between 5 and 20.");
            return;
        }

        const inputData = {
            bat_team: batTeam,
            bowl_team: bowlTeam,
            venue: venue,
            overs: overs,
            runs_last_5: runsLast5,
            wickets_last_5: wicketsLast5
        };
        
        resultBox.style.display = 'none';
        summaryText.innerHTML = "Calculating...";
        predictBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();

            if (response.ok) {
                summaryText.innerHTML = `
                    The **predicted final score** (Median) is **${Math.round(result.median)}** runs. 
                    The model estimates the score will likely fall between **${result.range_low}** and **${result.range_high}** runs.
                `;
                updateChart(result.bins, result.probs);
                resultBox.style.display = 'block';
            } else {
                let errorMessage = result.error || 'Prediction failed due to an unknown server error.';
                if (errorMessage.includes("unseen") || errorMessage.includes("not found")) {
                    errorMessage = "Error: A selected Team or Venue was not present in the model's training data.";
                }
                summaryText.innerHTML = `<span style="color: red;">❌ ${errorMessage}</span>`;
                resultBox.style.display = 'block';
            }

        } catch (error) {
            console.error('Fetch error:', error);
            summaryText.innerHTML = `<span style="color: red;">⚠️ Network Error: Could not connect to the Flask server.</span>`;
            resultBox.style.display = 'block';
        } finally {
            predictBtn.disabled = false;
        }
    }

    predictBtn.addEventListener('click', handlePrediction);


    function handleRefresh() {
        document.getElementById('overs').value = '';
        document.getElementById('runs_last_5').value = '';
        document.getElementById('wickets_last_5').value = '';

        document.getElementById('bat_team').selectedIndex = 0;
        document.getElementById('bowl_team').selectedIndex = 0;
        document.getElementById('venue').selectedIndex = 0;
        
        resultBox.style.display = 'none';
        summaryText.innerHTML = '';
        
        if (probabilityChart) {
            probabilityChart.destroy();
            probabilityChart = null;
        }
    }

    predictBtn.addEventListener('click', handlePrediction);
    refreshBtn.addEventListener('click', handleRefresh); 
});
