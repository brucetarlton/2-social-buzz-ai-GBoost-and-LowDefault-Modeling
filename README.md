
<br>
 
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]


<br>

# <p align="center">  Social [Buzz AI]()

<br><br>


<p align="center">
   <img src="https://github.com/user-attachments/assets/791a69e2-d09a-429f-9257-f6667fff5c04 ">
 </p>

<br><br>

[**Course:**]() Humanistic AI & Data Science (4th Semester)  
[**Institution:**]() PUC-SP  
**Professor:** [Erick Bacconi](https://www.linkedin.com/in/eric-bacconi-423137/)  



<br><br>


#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-%C2%B7%C2%B7%C2%B7%20Mindful%20AI%20Assistants%20%C2%B7%C2%B7%C2%B7-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)



<!--Confidentiality Statement-->

<br><br>


> [!IMPORTANT]
>
> ⚠️ Heads Up 
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
>
> * The course prioritizes [**hands-on practice**]() with real data in consulting scenarios.
>
> *  All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
>
> * [**Confidential information**]() from this repository remains private in [private repositories]().
>
>

#  

<br><br><br>

<!--End-->


<br><br>

> [!TIP]
>
> * [Access](): Workbook
> 
> * [Access](): 1- Code_GBM
>
> * [Access]():  2-Code_Low_Default_Models
>
>
> * [Access]():  Dataset Low_Default.csv
>

 
<br><br>


# Gradient Boosting Machines and Low-Default Modeling

A repository for research, implementation, and best practices with Gradient Boosting methods (GBM, XGBoost, LightGBM), H2O AutoML, and robust strategies for modeling extreme class imbalance ("Low Default") in data science for finance and risk.

<br><br>

## Table of Contents

- Comparative Table: GBM, XGBoost, LightGBM
- H2O Library for AutoML
- Gradient Boosting Algorithm Explained
- Strengths and Weaknesses
- XGBoost with Balanced Weights
- SMOTE + Random Forest for Low Default
- Modeling Strategies for Extreme Imbalance

<br><br>


## Comparative Table: GBM vs XGBoost vs LightGBM

<br>

| Feature | GBM  | XGBoost  | LightGBM  |
| :-- | :-- | :-- | :-- |
| Speed | Slower | Fast | Very Fast |
| Scalability | Limited | Good | Excellent |
| Core Method | Sequential decision trees | Optimized, parallel boosting | Leaf-wise split boosting |
| Memory Efficiency | Moderate | Sparse structure | High (big datasets) |
| Best Use Cases | Small data, flexibility | Large, structured data | Extensive feature sets, millions records |
| Handling Imbalanced Data | Needs weighting | `scale_pos_weight` | Parameter \& dataset-based |


<br><br>


## H2O Library for AutoML

H2O’s AutoML library runs a range of boosting, tree, and ensemble algorithms automatically, evaluating each under the same time and resource conditions, and selects the model with the best metrics. It supports advanced hyperparameter optimization and robust interpretability, highly effective for business and credit risk scenarios.

<br><br>

## Gradient Boosting Algorithm Explained

Gradient Boosting builds an ensemble by sequentially fitting new trees to the residuals (prediction errors) of previous trees. The ensemble learning objective is:


<br><br>


$$
F_{m}(x) = F_{m-1}(x) + \alpha h_{m}(x)
$$


<br><br>


- \$ F_{m}(x) \$: Prediction after m steps

- \$ h_{m}(x) \$: Weak learner/tree at step m

- \$ \alpha \$: Learning rate

<br><br>

At each step, a tree is fitted to minimize the gradient (loss function):

<br><br>

$$
w := w - \alpha \nabla J(w)
$$

<br><br>


### Stepwise GBM Flow

1. Begin with an initial prediction (mean, median, etc.)
2. Calculate residuals for all predictions
3. Fit a new tree to these residuals
4. Update aggregate predictions with controlled learning rate
5. Repeat until required accuracy or maximum trees



