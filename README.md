# Multimodal NFT Classifier

Web application for automated classification of NFT collections, using machine learning executed directly in the user's browser.

## Overview

The system allows classifying NFT collections by cross-referencing semantic data (texts) and statistical data (market metrics). The architecture was designed so that all neural network processing and feature extraction occur on the client side. This ensures privacy for the inputted data, eliminates backend server costs, and provides zero latency in classifications after the initial loading.

## Inference Modes

- **Inference with Description only:** Uses Natural Language Processing (NLP) to predict the category based solely on the collection's official textual description.
- **Inference with Description and Statistical Data:** Multimodal approach that combines description embeddings with 9 market variables (Total Volume, Total Sales, Supply, Owners, Average Price, Market Cap, Traits, Editors, and Floor Price) to maximize predictive accuracy.

## Technologies and Libraries (Web Application)

- **Interface:** HTML5, responsive CSS3, and Vanilla JavaScript.
- **Transformers.js:** Used to generate textual embeddings locally in the browser via the `all-MiniLM-L6-v2` model.
- **ONNX Runtime Web:** Inference engine responsible for loading and executing the neural network models in `.onnx` formats.

## How to Use

The application operates entirely online, requiring no local installation or environment configuration by the end user.

- Access the application's deployment link: `https://oscarwilliamm.github.io/nft-classification/`
- Wait for the secure download of the models to the browser's local cache. The status in the sidebar footer will change to **"Ready Models"**.
- Select the desired inference mode (text only or text + statistical data).
- Enter the NFT collection information into the text boxes and numerical inputs.
- Click on **Classify**. The mathematical calculation will be processed and displayed.
