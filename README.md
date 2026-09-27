# 🌾 Crop Yield Prediction System

An end-to-end Hybrid Machine Learning web application designed to predict agricultural crop yields (Tons / Hectare) using soil properties, climate variables, and fertilizer inputs.

---

## 🔗 Live Web Application
Evaluators can test the interactive system directly in any web browser without installing Python:
👉 **[Launch Live Web Demo](<https://crop-yield-predictionsystem-gdz37679bmz5jhtakkvzqc.streamlit.app/>)**

---

## 📌 Project Features
* **Hybrid Architecture:** Combines Random Forest Regression with Agronomic Guardrails to handle biological toxicity and environmental limits.
* **Interactive UI:** Dynamic Streamlit web application with real-time feedback and failure warnings.
* **Data Preprocessing:** Column Transformers with `OneHotEncoder` for handling categorical variables (State, District, Crop, Season, Soil Type).
* **Model Persistence:** Pipeline serialisation using `joblib`.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Regressor, Pipeline, ColumnTransformer)
* **Data Processing:** Pandas, NumPy
* **Model Export:** Joblib
* **Web UI / Deployment:** Streamlit, Streamlit Community Cloud

---
## 🏗️ Architecture & Mathematical Foundation

The system uses a **Hybrid Inference Engine** that combines a rule-based domain safety module with an ensemble machine learning pipeline. This architecture prevents standard decision-tree extrapolation errors in severe biological stress conditions.

```text
                    [ User Input Parameters ]
                                │
                                ▼
               ┌──────────────────────────────────┐
               │     Agronomic Safety Engine      │
               │   (Biological Threshold Rules)   │
               └────────────────┬─────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
[ Exceeds Biological Limits ]               [ Within Normal Limits ]
        │                                               │
        ▼                                               ▼
┌──────────────────────────┐               ┌──────────────────────────┐
│ Total Failure Diagnostic │               │    ColumnTransformer     │
│   Yield = 0.00 Tons/Ha   │               │  Preprocessing Pipeline  │
└──────────────────────────┘               └─────────────┬────────────┘
                                                         │
                                                         ▼
                                           ┌──────────────────────────┐
                                           │  Random Forest Regressor │
                                           │        Inference         │
                                           └─────────────┬────────────┘
                                                         │
                                                         ▼
                                            [ Predicted Yield (Tons/Ha) ]
```

### 1. Hybrid System Output Function

Let $X = \left[ \mathbf{x}\sb{\mathrm{cat}}, \mathbf{x}\sb{\mathrm{num}} \right]^T$ represent the complete input feature vector, where:

* $\mathbf{x}_{\mathrm{cat}} = \left[ \text{State, District, Crop, Season, Soil Type} \right]$
* $\mathbf{x}_{\mathrm{num}} = \left[ N, P, K, \mathrm{pH}, T, H, R, F \right]$

We define an agronomic indicator function $\mathbb{I}_{\mathrm{viable}}(X) \in \lbrace 0, 1 \rbrace$ based on domain threshold limits $L_c = \lbrace \mathrm{pH}_{\min}, \mathrm{pH}_{\max}, R_{\max}, T_{\min}, T_{\max} \rbrace$ for crop $c$:

$$
\mathbb{I}_{\text{viable}}(X) = \begin{cases} 0 & \text{if } \text{pH} \notin [\text{pH}_{\min}, \text{pH}_{\max}] \lor R > R_{\max} \lor T \notin [T_{\min}, T_{\max}] \lor \text{NutrientDepleted}(N, K, F) \\ 1 & \text{otherwise} \end{cases}
$$

The final predicted yield $\hat{Y}(X)$ is modeled as a piecewise step function:

$$
\hat{Y}(X) = \mathbb{I}_{\text{viable}}(X) \cdot f_{\text{RF}}(\Phi(X))
$$

Where $\Phi(X)$ is the feature transformation pipeline and $f_{\text{RF}}$ is the Random Forest regressor.

---

### 2. Preprocessing & Feature Encoding $\Phi(X)$
Categorical features $\mathbf{x}_{\text{cat}}$ are transformed using **One-Hot Encoding**:

$$
\Phi_{\text{OHE}}(x_j) = \mathbf{e}_k \in \{0, 1\}^K
$$

Where $K$ is the number of unique categories in feature $j$, and $\mathbf{e}_k$ is an indicator vector where entry $k=1$ for the active category and $0$ elsewhere.

Numerical features $\mathbf{x}_{\text{num}}$ are passed through directly, yielding the transformed vector:

$$
\Phi(X) = [\Phi_{\text{OHE}}(\mathbf{x}_{\text{cat}})^T, \mathbf{x}_{\text{num}}^T]^T
$$

---

### 3. Random Forest Regressor Formulation $f_{\text{RF}}$
The Random Forest model is an ensemble of $B$ independent decision trees $\{T_1, T_2, \dots, T_B\}$. The final output is the ensemble average of predictions across all trees:

$$
f_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
$$

Each individual tree $T_b(\mathbf{x})$ partitions the feature space into $M$ disjoint regions $R_1, R_2, \dots, R_M$:

$$
T_b(\mathbf{x}) = \sum_{m=1}^{M} c_m \cdot \mathbb{I}(\mathbf{x} \in R_m)
$$

Where $c_m$ is the mean target yield of training instances falling into terminal region $R_m$:

$$
c_m = \frac{1}{N_m} \sum_{\mathbf{x}_i \in R_m} y_i
$$

---

### 4. Node Splitting Criterion (MSE Minimization)
For a dataset node $Q$ containing $N_Q$ samples, decision trees select the feature $j$ and threshold $s$ that maximize impurity reduction $\Delta I(j, s)$:

$$
\Delta I(j, s) = \text{MSE}(Q) - \left( \frac{N_L}{N_Q} \text{MSE}(Q_L) + \frac{N_R}{N_Q} \text{MSE}(Q_R) \right)
$$

Where Mean Squared Error ($\text{MSE}$) for any node $Q$ is defined as:

$$
\text{MSE}(Q) = \frac{1}{N_Q} \sum_{i \in Q} (y_i - \bar{y}_Q)^2, \quad \bar{y}_Q = \frac{1}{N_Q} \sum_{i \in Q} y_i
$$

### 5. Ensemble Variance Reduction
Let $\sigma^2$ be the variance of an individual decision tree prediction, and $\rho$ be the correlation coefficient between any pair of trees. The overall variance of the Random Forest estimator is given by:

$$
\text{Var}(f_{\text{RF}}(\mathbf{x})) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
$$

By bagging random feature sub-samples (`max_features` = $\sqrt{p}$), $\rho$ is minimized, driving the second term towards $0$ as $B \to \infty$ and reducing model variance without increasing bias.

---

### 6. Model Evaluation Metrics

* **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

* **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$

* **Coefficient of Determination (R²):** `R² = 1 - [ Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)² ]`
---


## 💻 How to Run Locally

If you prefer to execute and test the code on your local environment:

### 1. Clone the Repository
```bash
git clone <https://github.com/Left-Crazy/Crop-Yield-Prediction_System.git>
cd "Crop Yeaild Prediction System"
