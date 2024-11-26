import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from xgboost import XGBClassifier

def main():
    st.set_page_config(page_title="Crime Data Analysis and Prediction", layout="wide")
    st.title('Crime Data Analysis and Prediction')

    # Sidebar for Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", ["Introduction", "Data Upload", "Exploratory Data Analysis", "Model Selection", "About"])

    if app_mode == "Introduction":
        st.markdown("""
            ## Welcome to the Crime Data Analysis and Prediction App!

            This app allows you to:
            - Upload your own dataset for analysis.
            - Explore the data with various visualizations.
            - Train and test different machine learning models.

            Use the sidebar to navigate through the different sections. Happy exploring!
        """)

    elif app_mode == "Data Upload":
        st.sidebar.success("Data Upload")
        st.subheader("Upload your dataset")

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(data.head())

            # Preprocess Data
            data.columns = data.columns.str.lower().str.replace(' ', '_')
            data.drop(columns=['incident_id', 'updated_at', 'created_at'], inplace=True, errors='ignore')
            #data.dropna(inplace=True)

            redundant_columns = ['parent_incident_type', 'state', 'city', 'location', 'census_tract', 'census_block',
                                 '2010_census_tract_', '2010_census_block_group', '2010_census_block', 'tractce20',
                                 'geoid20_tract', 'geoid20_blockgroup', 'geoid20_block']
            data.drop(columns=redundant_columns, inplace=True, errors='ignore')

            data['incident_datetime'] = pd.to_datetime(data['incident_datetime'], errors='coerce')
            data['incident_date'] = data['incident_datetime'].dt.date
            data['incident_time'] = data['incident_datetime'].dt.time
            data.drop(columns=['incident_datetime'], inplace=True, errors='ignore')

            data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
            data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
            data.dropna(subset=['latitude', 'longitude'], inplace=True)
            data['hour_of_day'] = data['hour_of_day'].astype(int)

            data['incident_type_primary'] = data['incident_type_primary'].str.lower()
            data['incident_description'] = data['incident_description'].str.lower()

            data['incident_date'] = pd.to_datetime(data['incident_date'], format='%Y-%m-%d', errors='coerce')
            data['year'] = data['incident_date'].dt.year
            data['month'] = data['incident_date'].dt.month
            data['day'] = data['incident_date'].dt.day

            def categorize_severity(incident_type_primary):
                if 'assault' in incident_type_primary or 'robbery' in incident_type_primary:
                    return 'High'
                elif 'theft' in incident_type_primary or 'burglary' in incident_type_primary:
                    return 'Medium'
                else:
                    return 'Low'

            data['severity'] = data['incident_type_primary'].apply(categorize_severity)
           # data = data[(data['latitude'].between(42.8, 43.0)) & (data['longitude'].between(-79.0, -78.7))]
            #data = data[data['year'] >= 2000]

            st.session_state['data'] = data

    elif app_mode == "Exploratory Data Analysis":
        st.sidebar.success("Exploratory Data Analysis")
        st.subheader("Exploratory Data Analysis")

        if 'data' in st.session_state:
            data = st.session_state['data']

            viz_option = st.selectbox('Select a visualization', [
                'Distribution by Hour of the Day', 'Incidents Over Months', 'Daily Trends by Day of Week',
                'Geographic Distribution by Neighborhood', 'Crime Incidents Heatmap', 'Count Incidents by Severity',
                'Incidents by Year and Month', 'Elbow Method for Optimal k', 'Crime Clusters', 'Top 10 Locations'
            ])

            if viz_option == 'Distribution by Hour of the Day':
                plt.figure(figsize=(10, 6))
                sns.countplot(x='hour_of_day', data=data)
                plt.title('Distribution of Incidents by Hour of the Day')
                st.pyplot(plt)

            elif viz_option == 'Incidents Over Months':
                monthly_counts = data['month'].value_counts().sort_index()
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=monthly_counts.index, y=monthly_counts.values)
                plt.title('Number of Incidents Over the Months')
                plt.xlabel('Month')
                plt.ylabel('Number of Incidents')
                st.pyplot(plt)

            elif viz_option == 'Daily Trends by Day of Week':
                plt.figure(figsize=(10, 6))
                sns.countplot(x='day_of_week', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                plt.title('Incidents by Day of the Week')
                st.pyplot(plt)

            elif viz_option == 'Geographic Distribution by Neighborhood':
                neighborhood_counts = data['neighborhood'].value_counts().head(10)
                plt.figure(figsize=(10, 6))
                sns.barplot(y=neighborhood_counts.index, x=neighborhood_counts.values, palette='viridis')
                plt.title('Incidents by Neighborhood')
                st.pyplot(plt)

            elif viz_option == 'Crime Incidents Heatmap':
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=data, x='longitude', y='latitude', fill=True, cmap='viridis', thresh=0, levels=100)
                plt.title('Heatmap of Crime Incidents')
                st.pyplot(plt)

            elif viz_option == 'Count Incidents by Severity':
                severity_counts = data['severity'].value_counts()
                plt.figure(figsize=(10, 6))
                sns.barplot(y=severity_counts.index, x=severity_counts.values, palette='viridis')
                plt.title('Count of Incidents by Severity')
                st.pyplot(plt)

            elif viz_option == 'Incidents by Year and Month':
                data['year_month'] = data['incident_date'].dt.to_period('M')
                year_month_counts = data['year_month'].value_counts().sort_index()
                plt.figure(figsize=(10, 6))
                year_month_counts.plot(kind='bar')
                plt.title('Incidents by Year and Month')
                plt.xlabel('Year-Month')
                plt.ylabel('Number of Incidents')
                st.pyplot(plt)

            elif viz_option == 'Elbow Method for Optimal k':
                coordinates = data[['latitude', 'longitude']]
                distortions = []
                K = range(1, 11)
                for k in K:
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(coordinates)
                    distortions.append(kmeans.inertia_)

                plt.figure(figsize=(10, 6))
                plt.plot(K, distortions, 'bx-')
                plt.xlabel('k')
                plt.ylabel('Distortion')
                plt.title('Elbow Method For Optimal k')
                st.pyplot(plt)

            elif viz_option == 'Crime Clusters':
                kmeans = KMeans(n_clusters=4)
                data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=data, palette='viridis')
                plt.title('Crime Clusters')
                st.pyplot(plt)
	    
            elif viz_option == 'Top 10 Locations':
                st.subheader('Top 10 Locations with the Most Incidents')
                top_locations = data['address'].value_counts().head(10)
                plt.figure(figsize=(10, 6))
                sns.barplot(y=top_locations.index, x=top_locations.values, palette='viridis')
                plt.title('Top 10 Locations with Most Incidents')
                plt.xlabel('Number of Incidents')
                plt.ylabel('Location')
                st.pyplot(plt)

        else:
            st.warning("Please upload a dataset in the Data Upload section.")

    elif app_mode == "Model Selection":
            st.sidebar.success("Model Selection")
            st.subheader("Model Training and Prediction")

            if 'data' in st.session_state:
                 data = st.session_state['data']

                # Feature and Target Selection
                 features = ['hour_of_day', 'latitude', 'longitude', 'year', 'month', 'day']
                 target = 'severity'

                 X = data[features]
                 y = LabelEncoder().fit_transform(data[target])

                # Split Data
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Model Selection
                 model_option = st.selectbox("Select a model", [
                    "Logistic Regression", "Naive Bayes", "Random Forest", "SVM", "XGBoost"
                ])

                 if model_option == "Logistic Regression":
                    model = LogisticRegression()
                    st.sidebar.subheader("Hyperparameter Tuning")
                    C = st.sidebar.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
                    model.set_params(C=C)

                 elif model_option == "Naive Bayes":
                    model = GaussianNB()

                 elif model_option == "Random Forest":
                    model = RandomForestClassifier()
                    st.sidebar.subheader("Hyperparameter Tuning")
                    n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
                    max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=20, value=10, step=1)
                    model.set_params(n_estimators=n_estimators, max_depth=max_depth)

                 elif model_option == "SVM":
                    model = SVC(probability=True)
                    st.sidebar.subheader("Hyperparameter Tuning")
                    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
                    C = st.sidebar.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
                    model.set_params(kernel=kernel, C=C)

                 elif model_option == "XGBoost":
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    st.sidebar.subheader("Hyperparameter Tuning")
                    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                    max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=15, value=6, step=1)
                    model.set_params(learning_rate=learning_rate, max_depth=max_depth)

                # Train Model
                 model.fit(X_train, y_train)
                 y_pred = model.predict(X_test)
                 accuracy = accuracy_score(y_test, y_pred)

                 st.write(f"### Accuracy: {accuracy:.2f}")

                # Classification Report
                 st.subheader("Classification Report")
                 st.text(classification_report(y_test, y_pred))
        
                # Confusion Matrix
                 st.subheader("Confusion Matrix")
                 cm = confusion_matrix(y_test, y_pred)
                 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
                 disp.plot(cmap='viridis', ax=plt.gca())
                 st.pyplot(plt)

                # Model-Specific Visualizations
                 if model_option == "Random Forest" or model_option == "XGBoost":
                    st.subheader("Feature Importance")
                    feature_importance = model.feature_importances_
                    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
                    plt.title('Feature Importance')
                    st.pyplot(plt)

                 # ROC Curve
                 if len(np.unique(y)) == 2:  # ROC only for binary classification
                    st.subheader("ROC Curve")
                    y_score = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(10, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
    
            else:
                 st.warning("Please upload and preprocess the dataset in the Data Upload section.")
    elif app_mode == "About":
         st.sidebar.success("About")
         st.markdown("""
            ## About This App
            - **Purpose:** This app demonstrates data analysis and machine learning models for crime datasets.
            - **Libraries Used:** Streamlit, pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
            - **Contact:** abhusara@buffalo.edu
         """)

if __name__ == "__main__":
    main()
