import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false; 

let embedder = null;
let sessionPhase1 = null;
let sessionPhase2 = null;
let categoryLabels = [];

window.switchTab = function(tabId) {
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
    
    // Esconde os resultados e porcentagens ao trocar de aba
    document.getElementById('result').style.display = 'none';
    document.getElementById('prob-details').style.display = 'none';
};

async function initializeModels() {
    try {
        const res = await fetch('models/categories.json');
        categoryLabels = await res.json();
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        sessionPhase1 = await ort.InferenceSession.create('models/model_phase1_v2.onnx');
        sessionPhase2 = await ort.InferenceSession.create('models/model_phase2_v2.onnx');
        document.getElementById('loading-status').innerText = 'Models Ready!';
        document.getElementById('btn-1').disabled = false;
        document.getElementById('btn-2').disabled = false;
        setTimeout(() => {document.getElementById('loading-status').style.display = 'none';}, 2000);
    } catch (error) {
        console.error("Error loading models:", error);
        document.getElementById('loading-status').innerText = 'Error loading AI models. Check console.';
    }
}

window.classify = async function(phase) {
    const resultBox = document.getElementById('result');
    const outputSpan = document.getElementById('category-output');
    
    outputSpan.innerText = "Analyzing...";
    resultBox.style.display = 'block';
    document.getElementById('prob-details').style.display = 'none';
    try {
        let text, float32Data, tensor, session;
        if (phase === 'phase1') {
            text = document.getElementById('desc-1').value;
            if (!text) return alert("Please enter a description.");
            session = sessionPhase1;
            const embOutput = await embedder(text, { pooling: 'mean', normalize: true });
            float32Data = embOutput.data; 
            tensor = new ort.Tensor('float32', float32Data, [1, 384]);
        } else {
            text = document.getElementById('desc-2').value;
            if (!text) return alert("Please enter a description.");
            session = sessionPhase2;
            const features = [
                parseFloat(document.getElementById('v-vol').value),
                parseFloat(document.getElementById('v-sales').value),
                parseFloat(document.getElementById('v-sup').value),
                parseFloat(document.getElementById('v-own').value),
                parseFloat(document.getElementById('v-avg').value),
                parseFloat(document.getElementById('v-mcap').value),
                parseFloat(document.getElementById('v-tr').value),
                parseFloat(document.getElementById('v-ed').value),
                parseFloat(document.getElementById('v-floor').value)
            ];

            const embOutput = await embedder(text, { pooling: 'mean', normalize: true });
            float32Data = new Float32Array(384 + 9);
            float32Data.set(embOutput.data, 0);
            float32Data.set(features, 384);
            tensor = new ort.Tensor('float32', float32Data, [1, 393]);
        }

        const inputName = session.inputNames[0];
        const labelOutputName = session.outputNames[0]; //categoria que mias se encaixa
        const probOutputName = session.outputNames[1];  //probabilidades
        console.log(`Enviando dados para a porta de entrada: ${inputName}`);
        const feeds = {};
        feeds[inputName] = tensor;
        const results = await session.run(feeds);
        console.log(`Lendo dados das portas de saída: ${labelOutputName} e ${probOutputName}`);
        const predictedIndex = Number(results[labelOutputName].data[0]);
        const predictedCategory = categoryLabels[predictedIndex];
        const probabilities = results[probOutputName].data;
        const mainConfidence = (probabilities[predictedIndex] * 100).toFixed(2);
        outputSpan.innerText = `${predictedCategory} (${mainConfidence}%)`;
        const probListElement = document.getElementById('prob-list');
        probListElement.innerHTML = ''; 
        
        let probArray = [];
        for (let i = 0; i < categoryLabels.length; i++) {
            probArray.push({
                category: categoryLabels[i],
                percent: (probabilities[i] * 100)
            });
        }
 
        probArray.sort((a, b) => b.percent - a.percent);
        probArray.forEach(item => {
            const li = document.createElement('li');
            li.style.marginBottom = "0.5rem";
            li.style.borderBottom = "1px solid #f0f0f0";
            li.style.paddingBottom = "0.2rem";
            li.innerHTML = `<strong>${item.category}:</strong> <span style="float:right;">${item.percent.toFixed(2)}%</span>`;
            probListElement.appendChild(li);
        });
        document.getElementById('prob-details').style.display = 'block';
    } catch (error) {
        console.error("Inference error:", error);
        outputSpan.innerText = "Error in prediction. Check Console (F12).";
        document.getElementById('prob-details').style.display = 'none';
    }
};

initializeModels();
