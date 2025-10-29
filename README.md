# NexGen Predictive Delivery Optimizer

## 1. Project Overview
This project was developed as part of the NexGen Logistics Innovation Challenge.  
The goal was to analyze operational data to understand the causes of delivery delays and develop a predictive tool to identify potential delays before they occur.

---

## 2. Problem Statement
NexGen Logistics manages 200+ monthly orders, 5 warehouses, and a fleet of 50 vehicles across India and internationally.  
They face ongoing challenges such as:
- Frequent delivery delays and quality issues  
- Rising operational costs  
- Limited data-driven planning and forecasting capabilities  

The objective was to design a system that combines data visualization and machine learning to support better logistics decision-making.

---

## 3. Approach
1. **Data Exploration** – Analyzed multiple datasets including delivery performance, orders, routes, costs, and customer feedback.  
2. **Data Preparation** – Cleaned and merged data for consistency across order IDs and carrier information.  
3. **Dashboard Design** – Built an interactive **Streamlit + Plotly** dashboard for visual analysis and operational insights.  
4. **Predictive Modeling** – Developed a machine learning model to predict delivery delays using simplified and interpretable features such as:
   - Carrier  
   - Promised Delivery Days  
   - Route  
   - Weather Impact  

---

## 4. Machine Learning Model
A classification model was trained using historical delivery data to estimate the likelihood of a delay.  
The model provides an early risk signal before dispatch, enabling proactive actions.

Due to limited dataset size, the model accuracy is moderate. The next steps include:
- Collecting more historical data for training  
- Incorporating additional operational variables  
- Improving interpretability and explainability of predictions  
- Integrating mathematical and historical optimization for real-time decision-making  

---

## 5. Tools and Technologies
- Python (pandas, numpy, scikit-learn, xgboost)  
- Streamlit for dashboard development  
- Plotly for data visualization  
- Joblib for model persistence  

