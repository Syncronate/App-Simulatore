// --- 1. Impostazioni Iniziali e Dimensioni del Grafico ---
const margin = { top: 50, right: 50, bottom: 50, left: 50 };
const width = 800 - margin.left - margin.right;
const height = 500 - margin.top - margin.bottom;

// Creiamo l'elemento SVG che conterrà il grafico
const svg = d3.select("#grafico-container")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// --- 2. Caricamento e Parsing dei Dati ---
d3.csv("dati_idrometrici.csv").then(data => {

    // Convertiamo i dati nel formato corretto
    const parseTime = d3.timeParse("%d/%m/%Y %H:%M");
    data.forEach(d => {
        d.Timestamp = parseTime(d.Timestamp);
        d.Q10 = +d.Q10;
        d.Q50 = +d.Q50;
        d.Q90 = +d.Q90;
    });

    // --- 3. Definizione delle Scale (Assi X e Y) ---
    
    // Scala X per il tempo
    const xScale = d3.scaleTime()
        .domain(d3.extent(data, d => d.Timestamp))
        .range([0, width]);

    // Scala Y per il livello idrometrico
    const yScale = d3.scaleLinear()
        .domain([
            d3.min(data, d => d.Q10) - 0.05, // Aggiungiamo un po' di margine
            d3.max(data, d => d.Q90) + 0.05
        ])
        .range([height, 0]);

    // --- 4. Disegno degli Assi ---
    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat("%H:%M")));

    svg.append("g")
        .call(d3.axisLeft(yScale))
        .append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left + 15)
        .attr("x", -height / 2)
        .attr("text-anchor", "middle")
        .text("Livello Idrometrico [m]");

    // --- 5. Disegno delle Linee dei Percentili ---
    const createLine = (yValue) => d3.line()
        .x(d => xScale(d.Timestamp))
        .y(d => yScale(d[yValue]));

    svg.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("d", createLine("Q10"))
        .style("stroke", "#64b5f6"); // Blu chiaro

    svg.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("d", createLine("Q50"))
        .style("stroke", "#1976d2"); // Blu medio

    svg.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("d", createLine("Q90"))
        .style("stroke", "#0d47a1"); // Blu scuro

    // Aggiungiamo una legenda
    const legend = svg.append("g").attr("transform", `translate(${width - 100}, 0)`);
    legend.append("rect").attr("x", 0).attr("y", 0).attr("width", 10).attr("height", 10).style("fill", "#64b5f6");
    legend.append("text").attr("x", 15).attr("y", 10).text("Q10").style("font-size", "12px");
    legend.append("rect").attr("x", 0).attr("y", 20).attr("width", 10).attr("height", 10).style("fill", "#1976d2");
    legend.append("text").attr("x", 15).attr("y", 30).text("Q50").style("font-size", "12px");
    legend.append("rect").attr("x", 0).attr("y", 40).attr("width", 10).attr("height", 10).style("fill", "#0d47a1");
    legend.append("text").attr("x", 15).attr("y", 50).text("Q90").style("font-size", "12px");


    // --- 6. Animazione dell'Acqua (Effetto Fluido) ---
    
    // Creiamo un'area che rappresenterà l'acqua, basata sulla previsione Q50
    const waterArea = d3.area()
        .x(d => xScale(d.Timestamp))
        .y0(height) // La base dell'area è il fondo del grafico
        .y1(d => yScale(d.Q50)); // La parte superiore segue la linea Q50

    // Aggiungiamo il path per l'acqua
    const waterPath = svg.append("path")
        .datum(data)
        .attr("fill", "rgba(25, 118, 210, 0.5)") // Colore blu semi-trasparente
        .attr("d", waterArea);

    // Funzione per l'animazione delle onde
    function animateWave() {
        waterPath
            .transition()
            .duration(2000) // Durata dell'animazione
            .ease(d3.easeSinInOut) // Effetto di "respiro"
            .attrTween("d", function() {
                // Interpola tra due stati dell'onda per creare movimento
                const originalPath = waterArea(data);
                const wavePath = d3.area()
                    .x(d => xScale(d.Timestamp))
                    .y0(height)
                    .y1(d => yScale(d.Q50 + Math.sin(xScale(d.Timestamp) / 50) * 0.01)) // Aggiunge una piccola onda sinusoidale
                    (data);

                return d3.interpolatePath(originalPath, wavePath);
            })
            .transition()
            .duration(2000)
            .ease(d3.easeSinInOut)
            .attrTween("d", function() {
                 const wavePath = d3.area()
                    .x(d => xScale(d.Timestamp))
                    .y0(height)
                    .y1(d => yScale(d.Q50 + Math.sin(xScale(d.Timestamp) / 50) * 0.01))
                    (data);
                const originalPath = waterArea(data);
                return d3.interpolatePath(wavePath, originalPath);
            })
            .on("end", animateWave); // Ripete l'animazione all'infinito
    }

    // Avviamo l'animazione
    animateWave();

}).catch(error => {
    console.error("Errore nel caricamento dei dati:", error);
});