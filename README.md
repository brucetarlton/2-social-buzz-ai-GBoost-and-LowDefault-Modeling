
<br>
 
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]


<br>

# <p align="center">  Social [Buzz AI]() - Gradient Boosting Machines and Low-Default Modeling

<br>

A repository for research, implementation, and best practices with Gradient Boosting methods (GBM, XGBoost, LightGBM), H2O AutoML, and robust strategies for modeling extreme class imbalance ("Low Default") in data science for finance and risk.

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
> * [Access](https://github.com/Mindful-AI-Assistants/2-social-buzz-ai-GBoost-and-LowDefault-Modeling/blob/f699dd62c269f0d002615a9cecc7f0351fb4fb7c/Workbook_GBoost/GBoost.pdf): Workbook
> 
> * [Access](https://github.com/Mindful-AI-Assistants/2-social-buzz-ai-GBoost-and-LowDefault-Modeling/blob/5e5dedeb27e2d3631464dc5aa07c93b311576a55/Code_GBM/GBM_.ipynb): 1- Code_GBM
>
> * [Access](https://github.com/Mindful-AI-Assistants/2-social-buzz-ai-GBoost-and-LowDefault-Modeling/blob/5e5dedeb27e2d3631464dc5aa07c93b311576a55/Code_Low_Default_Models/Low_Default_Models.ipynb):  2-Code_Low_Default_Models
>
>
> * [Access](https://github.com/Mindful-AI-Assistants/2-social-buzz-ai-GBoost-and-LowDefault-Modeling/tree/5e5dedeb27e2d3631464dc5aa07c93b311576a55/Dataset_Low_DeFault):  Dataset Low_Default.csv
>

 
<br><br><br>


#

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
\Huge
F_{m}(x) = F_{m-1}(x) + \alpha h_{m}(x)
$$


<br><br>


- $\Huge {m}(x)\$: Prediction after m steps

- $\Huge h_{m}(x)\$: Weak learner/tree at step m

- $\Huge \alpha\$: Learning rate

<br><br>

At each step, a tree is fitted to minimize the gradient (loss function):


<br><br>

$$
\Huge
w := w - \alpha \nabla J(w)
$$

<br><br>


### Stepwise GBM Flow

1. Begin with an initial prediction (mean, median, etc.)
2. Calculate residuals for all predictions
3. Fit a new tree to these residuals
4. Update aggregate predictions with controlled learning rate
5. Repeat until required accuracy or maximum trees


<br><br>


## Strengths and Weaknesses

<br>

### Strengths

- **Accuracy:** Sequential error correction produces highly precise models
- **Flexibility:** Can handle various data types, losses, and objectives
- **Regularization:** Learning rate, subsampling, and tree depth reduce overfitting

<br>

### Weaknesses

- **Training Time:** Slow for very large datasets
- **Tuning Required:** Sensitive to learning rate, tree depth, and trees count


<br><br>


## XGBoost with Balanced Weights

XGBoost enhances GBM by aggressive optimization, parallel processing, and support for imbalanced data with the `scale_pos_weight` parameter. Weighted loss function example:

<br>

$$
\Huge
Loss = \sum_{i=1}^{N} w_i \cdot l(y_i, f(x_i))
$$


<br>

### where $\Huge w_i\$ prioritizes rare/default cases.


<br>

### Advantages

- Superior for tabular, unstructured, or highly imbalanced data
- Extremely robust, fast, and captures complex data patterns
- Enables cross-validation and advanced tuning options

<br>

### Disadvantages

- Needs some positive/default event cases to operate
- If event rate <<0.1%, may not suffice alone
- May overfit if weights are extreme and not tuned with care

<br><br>


## LightGBM and Leaf-wise Boosting

LightGBM uses leaf-wise growth rather than level-wise, dramatically increasing performance on very large, multi-feature datasets. It retains speed even as data size scales into the millions of rows.

<br><br>


## SMOTE + Random Forest for Low Default

- **SMOTE:** Synthesizes extra default (minority class) data using nearest neighbors, sharply balancing class ratios
- **Random Forest:** Trains on synthetic-balanced data, offering stability and high robustness under correlation and noise
- **Prediction:** Final RF model is applied to the real input features for risk prediction



### Key SMOTE+RF Formula

Synthetic instance generation:

<br><br>

$$
\Huge
\mathbf{x}_{\text{new}} = \mathbf{x}_i + \delta \cdot (\mathbf{x}_{nn} - \mathbf{x}_i)
$$


<br><br>

 
<!-- 
 where $\mathbf{x}_i\$ is a minority instance, $\Huge \mathbf{x}_{nn}$ a neighbor, and $\Huge \delta \in\$ random.
-->


### [Where]():  `x_i` is a minority instance, `x_nn` a neighbor, and `δ ∈ random`




<br>

### SMOTE + RF: Pros and Cons

<br>

**Advantages**

- Directly fixes imbalance by creating synthetic defaults
- RF is robust to feature noise and correlation
- Practical, easy, and works well in most real low-default cases

<br>

**Disadvantages**

- Synthetic data may cause overfitting or be unrealistic with very rare events
- Not effective when only 2-3 true defaults exist
- The synthetic process can distort the target distribution

<br>

## Modeling Strategies for Extreme Imbalance

- Combine sampling (SMOTE), weighting, and boosting for best results in low-default settings
- Use cross-validation and specialized metrics (ROC, recall, F1) to monitor real performance on rare-event detection

<br><br>

## Reference


[1](): https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2013.00021/full

[2](): http://uc-r.github.io/gbm_regression


[4](): https://www.digitalocean.com/community/tutorials/gradient-boosting-for-classification

[5](): https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html

[6](): https://www.geeksforgeeks.org/machine-learning/ml-gradient-boosting/

[7](): [GBoost TEDE - PUCSP]()


<br><br>



## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)


#### [Contact and Support]()

- For notebook files, detailed tutorials, or enhanced visualizations, please reach out.
- Interested in Python notebooks simulating these dynamics or advanced Humanistic AI models? Just ask!

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />



<br>



<p align="center">  ────────────── 🔭⋆ ──────────────


<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>


<b><br>

#

##### <p align="center">Copyright 2025 Mindful-AI-Assistants. Code released under the  [MIT license.](https://github.com/Mindful-AI-Assistants/lacan-psychoanalysis-math-graphs/blob/28d9178584b831679dec129fb0aa040203ce0e9e/LICENSE.md)





