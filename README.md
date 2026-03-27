# farmacia_etica_portal
A centralized, secure, and fully automated web portal built to drive data-informed decisions for Farmacia Etica. This repository houses three advanced Machine Learning and Optimization applications wrapped in a unified Streamlit multi-page interface, backed by cloud storage and a custom authentication system.

🔴 Farmacia Etica - AI & Data Analytics Portal
A centralized, secure, and fully automated web portal built to drive data-informed decisions for Farmacia Etica. This repository houses three advanced Machine Learning and Optimization applications wrapped in a unified Streamlit multi-page interface, backed by cloud storage and a custom authentication system.

🌟 Key Features
1. Unified Security & Infrastructure (Inicio.py)
Custom Authentication: A built-in user management system featuring SHA-256 password hashing.

Automated Admin Approvals: SMTP email integration to notify administrators of new sign-ups and alert users upon approval.

AWS S3 Integration: Seamlessly reads and writes datasets and user credentials to the cloud using boto3, ensuring the app remains stateless and highly available.

2. 🔮 Order Forecasting (1_Prediccion_Ordenes.py)
A deep learning time-series forecasting tool designed to predict inventory demand for the upcoming 3 weeks.

Hybrid Neural Network: Built with PyTorch, the model utilizes a Conv1D layer for feature extraction, an LSTM for long-term memory, and an RNN for temporal compression.

Clustered Training (MoE): Products are dynamically grouped using PCA and K-Means clustering, allowing specialized neural networks to train on distinct market behaviors.

Automated Hyperparameter Tuning: Integrates Optuna (TPE Sampler) with custom early-stopping callbacks to automatically find the optimal neural network architecture, learning rate, and batch size without getting trapped in local minima.

3. 📈 Molecule Market Analysis (3_Analisis_Moleculas.py)
An analytical engine that evaluates pharmaceutical import data to assess market trends and risks.

Statistical Feature Engineering: Automatically calculates rolling averages, trailing zeros, Gini coefficients (market share distribution), and linear regression slopes/p-values for specific active ingredients.

Custom CLPSO Algorithm: Uses Comprehensive Learning Particle Swarm Optimization (opt.py) with a custom Fisher-style objective function to cleanly separate and classify high-risk vs. stable market molecules.

4. 📊 Price Optimization (2_Calculadora_Precio.py)
A dynamic pricing calculator designed to find the optimal profit margins.

Bayesian Optimization: Utilizes bayes_opt to mathematically maximize target multipliers based on market ratios, adjusting mathematical curve parameters (a, b, gamma) to find the perfect pricing formula.
