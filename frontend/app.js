// frontend/app.js

async function predictGenre() {

    const title = document.getElementById("title").value;

    const plot = document.getElementById("plot").value;

    const year = parseInt(document.getElementById("year").value);

    const resultsDiv = document.getElementById("results");

    resultsDiv.style.display = "block";

    resultsDiv.innerHTML = "<p>Predicting genres...</p>";

    try {

        // CAMBIA ESTA URL POR LA DE RENDER
        const response = await fetch("https://movie-genre-classifier-api.onrender.com/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                title,
                plot,
                year
            })
        });

        const data = await response.json();

        let html = `
            <h2>Top Predictions for "${data.movie_title}"</h2>
            <br>
        `;

        data.top_predictions.forEach(item => {

            html += `
                <div class="genre-item">

                    <span class="genre-name">
                        ${item[0]}
                    </span>

                    <span class="genre-score">
                        ${(item[1] * 100).toFixed(2)}%
                    </span>

                </div>
            `;
        });

        resultsDiv.innerHTML = html;

    } catch (error) {

        resultsDiv.innerHTML = `
            <p style="color:red;">
                Error connecting to API
            </p>
        `;

        console.error(error);
    }
}
