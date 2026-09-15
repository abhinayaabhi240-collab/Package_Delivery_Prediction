# Package_Delivery_Prediction
Package Delivery Time Prediction using Machine Learning and Streamlit. Predict delivery ETA based on distance, package weight, traffic, weather, vehicle type, warehouse delay, handling time, and priority.
#  Package Delivery Time Prediction using Streamlit

A machine learning web application that predicts **package delivery time** using shipment, route, traffic, weather, vehicle, and logistics-related information.

The application is built using **Python, Scikit-learn, and Streamlit** and provides both individual shipment ETA prediction and batch prediction through CSV files.

## Features

*  Predict delivery time for an individual package
*  Display estimated delivery time in hours and days
*  Consider traffic conditions
*  Consider weather conditions
*  Support Bike, Van, and Truck vehicles
*  Support Standard and Express delivery priorities
*  Consider warehouse delay and package handling time
*  Upload CSV files for batch shipment prediction
*  View model feature importance
*  Download predicted shipment results as CSV
*  Provides dispatch/SLA recommendations
*  Includes quick shipment presets for different delivery scenarios

## Machine Learning Model

The project uses a **Random Forest Regressor** to predict package delivery time in hours.

### Model Configuration

```text
Algorithm: RandomForestRegressor
Number of Trees: 200
Random State: 42
```

### Preprocessing

Categorical features are converted using **OneHotEncoder**, while numerical features are passed directly to the model.

**Categorical Features:**

* Traffic Level
* Weather
* Vehicle Type
* Priority

**Numerical Features:**

* Distance
* Package Weight
* Warehouse Delay
* Handling Time

##  Model Performance

The model is evaluated using an 80/20 train-test split.

| Metric   |      Result |
| -------- | ----------: |
| R² Score |        0.85 |
| MAE      | ~1.53 hours |
| RMSE     | ~1.86 hours |

These results are based on the included synthetic dataset and the project's fixed train/test split.

##  Input Features

The model accepts the following shipment information:

| Feature                 | Description                     |
| ----------------------- | ------------------------------- |
| `distance_km`           | Delivery distance in kilometers |
| `package_weight_kg`     | Package weight in kilograms     |
| `traffic_level`         | Low, Medium, or High            |
| `weather`               | Clear, Rain, or Storm           |
| `vehicle_type`          | Bike, Van, or Truck             |
| `warehouse_delay_hours` | Delay at the warehouse          |
| `handling_time_hours`   | Sorting and handling time       |
| `priority`              | Standard or Express             |

### Target Variable

```text
delivery_time_hours
```

The target represents the actual delivery time in hours in the training dataset.

##  Application Sections

### 1. Shipment ETA Calculator

Enter package and route details to calculate the estimated delivery time.

The application displays:

* Estimated transit time in hours
* Approximate duration in days
* Important transit delay factors
* Dispatch recommendation
* Raw input features

It also includes presets such as:

*  Express Van Delivery
*  Heavy Freight Long-Haul
*  Local Courier Bike

### 2. Manifest Batch Screening

Upload a CSV file containing multiple shipments.

The application can:

* Predict ETA for every shipment
* Show total shipments
* Calculate average ETA
* Show fastest delivery
* Show longest delivery
* Filter results by vehicle and priority
* Download the scored manifest as a CSV file

### 3. Feature Importance & Diagnostics

The application displays:

* Machine learning algorithm
* Preprocessing architecture
* Model evaluation metrics
* Feature importance visualization

##  Project Structure

```text
Package_Delivery_Prediction_Streamlit/
│
├── app.py
├── train_model.py
├── requirements.txt
├── package_delivery_model.pkl
├── feature_importance.png
│
└── data/
    └── package_delivery_data.csv
```

##  Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Package_Delivery_Prediction_Streamlit
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

##  Train the Model

To train the Random Forest model again:

```bash
python train_model.py
```

This generates:

```text
package_delivery_model.pkl
feature_importance.png
```

##  Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

##  Requirements

The project uses:

```text
pandas
scikit-learn
joblib
matplotlib
streamlit
```

Install them with:

```bash
pip install -r requirements.txt
```

##  Dataset

The project includes a **synthetic package delivery dataset containing 500 shipment records**.

The dataset includes information about:

* Delivery distance
* Package weight
* Traffic
* Weather
* Vehicle type
* Warehouse delay
* Handling time
* Delivery priority
* Actual delivery time

> **Note:** The dataset is synthetic and intended for educational, demonstration, and machine-learning practice purposes.

##  Technologies Used

* **Python**
* **Pandas** – Data processing
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning
* **Random Forest Regressor** – Prediction model
* **Joblib** – Model saving/loading
* **Matplotlib** – Feature importance visualization
* **Streamlit** – Interactive web application

##  Project Objective

The main objective of this project is to demonstrate how machine learning can be applied to **logistics and last-mile delivery prediction**.

By considering different shipment and environmental factors, the system provides an estimated delivery time that can help with:

* Delivery planning
* Shipment monitoring
* Fleet management
* ETA estimation
* Logistics analysis

##  Future Improvements

* Add real-time traffic data
* Integrate GPS/location information
* Use real-world delivery datasets
* Add delivery-time trends and charts
* Compare multiple machine-learning algorithms
* Deploy the application online
* Add authentication and shipment tracking
* Improve model performance with hyperparameter tuning

