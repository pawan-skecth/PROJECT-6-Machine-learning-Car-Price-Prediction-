# PROJECT-6-Machine-learning-Car-Price-Prediction

This project focuses on predicting the **selling price of used cars** based on various features like **brand, year of manufacture, fuel type, and kilometers driven**. It involves **data preprocessing, exploratory data analysis (EDA), outlier detection, machine learning model training, and deployment using Gradio**.

---

## **Introduction**  
The used car market is growing rapidly, and pricing a used vehicle correctly is a significant challenge for both buyers and sellers. Traditional pricing methods rely on subjective judgments, making them inconsistent. **Machine Learning (ML) offers a data-driven approach** to estimate the fair price of a used car based on historical sales data.  

This project aims to:
1. **Analyze a dataset of used car sales.**  
2. **Clean and preprocess the data to handle inconsistencies.**  
3. **Perform exploratory data analysis (EDA) to understand data distributions.**  
4. **Train a regression model to predict car prices.**  
5. **Deploy the model as a web application using Gradio.**  

---

## **1. Data Preprocessing and Cleaning**  

The dataset (`quikr_car.csv`) contains information about different used cars, including features like:
- **Car Name** (e.g., "Honda City VX")  
- **Company/Brand** (e.g., "Honda")  
- **Year of Manufacture** (e.g., 2015)  
- **Kilometers Driven** (e.g., "50,000 kms")  
- **Fuel Type** (e.g., Petrol, Diesel, CNG)  
- **Price** (e.g., ₹5,00,000 or "Ask For Price")  

### **Steps in Data Preprocessing:**  

1. **Checking Data Structure:**  
   - `df.info()` gives an overview of column types and missing values.  
   - `df.describe()` provides statistical summaries.  
   - `df.duplicated().sum()` identifies duplicate records, which are removed.  

2. **Handling Missing and Inconsistent Data:**  
   - The **"kms_driven"** column contains text ("50,000 kms"), so we remove `" kms"` and convert it to numeric format.  
   - Some `kms_driven` values mistakenly contain "Petrol" instead of a number. These rows are removed.  
   - **Price column** includes values like `"Ask For Price"` which are removed before converting the column to float.  
   - **Missing values in `fuel_type`** are replaced with `"Missing"`.  
   - `year` is converted to an integer after handling non-numeric values.  

3. **Outlier Detection and Handling:**  
   - Extreme outliers in `Price` (cars priced above ₹60 lakh) are removed.  
   - **IQR (Interquartile Range) method** is used to handle outliers in `kms_driven`. Values beyond 1.5 * IQR are replaced with the median.  

4. **Feature Engineering:**  
   - The **"Car Name"** is simplified by keeping only the first three words (e.g., `"Honda City VX"` → `"Honda City VX"`).  

---

## **2. Exploratory Data Analysis (EDA)**  
EDA helps us understand the dataset’s characteristics before building a model.  

1. **Distribution of Car Prices** – A histogram is plotted to visualize how car prices are distributed.  
2. **Cars by Manufacturing Year** – A bar chart shows the number of cars available for each year.  
3. **Fuel Type Distribution** – A pie chart illustrates the percentage of cars by fuel type.  
4. **Kilometers Driven Distribution** – A box plot shows how the distance traveled varies across different cars.  
5. **Top 10 Car Brands** – A bar chart displays the most common brands in the dataset.  
6. **Fuel Type vs. Price** – A box plot visualizes how car prices vary by fuel type.  
7. **Checking Normality** – Histogram and Q-Q plots help us determine if the numerical features follow a normal distribution.  

---

## **3. Model Building**  
A **machine learning model** is trained to predict car prices based on various input features.

### **Feature Selection:**  
- **Categorical Features:** `name`, `company`, `fuel_type`  
- **Numerical Features:** `kms_driven`, `year`  
- **Target Variable:** `Price`  

### **Data Preprocessing for Model Training:**  
1. **One-Hot Encoding** is applied to categorical features (`name`, `company`, `fuel_type`).  
2. **MinMaxScaler** is used to scale numerical features (`kms_driven`, `year`).  
3. **Log Transformation of Target (`Price`)**: Since car prices vary widely, applying `np.log1p(Price)` makes the distribution more normal, improving model performance.  

### **Model Training and Evaluation:**  
- The dataset is split into **80% training data and 20% test data**.  
- A **Linear Regression Model** is trained using a **Pipeline**, which combines preprocessing and model training.  
- Model performance is evaluated using:
  - **Mean Squared Error (MSE)** – Measures average squared error.  
  - **R² Score** – Indicates how well the model explains variance in prices.  

---

## **4. Model Deployment with Gradio**  
To make the model accessible to users, it is deployed as a **web application using Gradio**.  

### **Steps in Deployment:**  
1. **Save the trained model using `pickle`**:  
   ```python
   import pickle
   with open("car_price_model.pkl", "wb") as file:
       pickle.dump(model_pipeline, file)
   ```
2. **Create a Gradio Interface:**  
   - Users enter the car name, company, year, kilometers driven, and fuel type.  
   - The model predicts the car’s estimated price.  
   - The **log-transformed prediction is converted back** using `np.expm1()`.  

### **User Input Fields:**  
- **Car Name** (Textbox)  
- **Company** (Textbox)  
- **Year** (Number Input)  
- **Kilometers Driven** (Number Input)  
- **Fuel Type** (Dropdown: Petrol, Diesel, CNG, Electric)  

### **Predicted Output:**  
- Displays **"Estimated Price: ₹ X,XXX,XXX"**  


## **Conclusion**  
This **Car Price Prediction Project** successfully implements **data cleaning, EDA, feature engineering, machine learning model training, and web deployment**. The **Gradio interface** allows users to interact with the model and estimate car prices easily.

### **Key Achievements:**
✅ **Preprocessed real-world car sales data** by handling missing values, outliers, and text-based inconsistencies.  
✅ **Explored data visually** to identify trends in car prices.  
✅ **Trained a Linear Regression Model** with **log transformation** for better price predictions.  
✅ **Built a user-friendly web app** for real-time price estimation.  

This project can be extended further by:
- Trying **advanced models** like Random Forest, XGBoost, or Neural Networks.  
- Collecting **newer car sales data** for better predictions.  
- Adding **more features** like location, car condition, and transmission type.  

This **Machine Learning solution** provides a **data-driven approach to car valuation**, helping users make informed decisions when buying or selling used cars. 🚗💰
