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
    st.title('Crime Data Analysis and Prediction')

    # Upload Dataset
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Preprocess Data
        data.columns = data.columns.str.lower().str.replace(' ', '_')
        data.drop(columns=['incident_id', 'updated_at', 'created_at'], inplace=True)
        data.dropna(inplace=True)
        
        redundant_columns = ['parent_incident_type', 'state', 'city', 'location', 'census_tract', 'census_block',
                             '2010_census_tract_', '2010_census_block_group', '2010_census_block', 'tractce20',
                             'geoid20_tract', 'geoid20_blockgroup', 'geoid20_block']
        data.drop(columns=redundant_columns, inplace=True)
        
        data['incident_datetime'] = pd.to_datetime(data['incident_datetime'], errors='coerce')
        data['incident_date'] = data['incident_datetime'].dt.date
        data['incident_time'] = data['incident_datetime'].dt.time
        data.drop(columns=['incident_datetime'], inplace=True)
        
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data.dropna(subset=['latitude', 'longitude'], inplace=True)
        data['hour_of_day'] = data['hour_of_day'].astype(int)
        
        data['incident_type_primary'] = data['incident_type_primary'].str.lower()
        data['incident_description'] = data['incident_description'].str.lower()
        
        data['incident_date'] = pd.to_datetime(data['incident_date'], format='%m/%d/%Y', errors='coerce')
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
        data = data[(data['latitude'].between(42.8, 43.0)) & (data['longitude'].between(-79.0, -78.7))]
        data = data[data['year'] >= 2000]

        # Show Data
        st.write(data.head())

        # Visualization Options
        st.subheader('Visualization Options')
        viz_option = st.selectbox('Select a visualization', [
            'Distribution by Hour of the Day', 'Incidents Over Months', 'Daily Trends by Day of Week',
            'Geographic Distribution by Neighborhood', 'Crime Incidents Heatmap', 'Count Incidents by Severity',
            'Incidents by Year and Month', 'Elbow Method for Optimal k', 'Crime Clusters', 'Top 10 Locations'
        ])

        if viz_option == 'Distribution by Hour of the Day':
            # Visualize the distribution of incidents by hour of the day.
            st.subheader('Distribution of Incidents by Hour of the Day')
            plt.figure(figsize=(10, 6))
            sns.countplot(x='hour_of_day', data=data)
            plt.title('Distribution of Incidents by Hour of the Day')
            st.pyplot(plt)

        elif viz_option == 'Incidents Over Months':
            # Plot the number of incidents over the months
            st.subheader('Number of Incidents Over the Months')
            monthly_counts = data['month'].value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=monthly_counts.index, y=monthly_counts.values)
            plt.title('Number of Incidents Over the Months')
            plt.xlabel('Month')
            plt.ylabel('Number of Incidents')
            st.pyplot(plt)

        elif viz_option == 'Daily Trends by Day of Week':
            # Daily Trends by Day of Week
            st.subheader('Daily Trends by Day of the Week')
            plt.figure(figsize=(10, 6))
            sns.countplot(x='day_of_week', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            plt.title('Incidents by Day of the Week')
            st.pyplot(plt)

        elif viz_option == 'Geographic Distribution by Neighborhood':
            # Geographic Distribution by Neighborhood
            st.subheader('Geographic Distribution by Neighborhood')
            neighborhood_counts = data['neighborhood'].value_counts().head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(y=neighborhood_counts.index, x=neighborhood_counts.values, palette='viridis')
            plt.title('Incidents by Neighborhood')
            plt.xlabel('Number of Incidents')
            plt.ylabel('Neighborhood')
            st.pyplot(plt)

        elif viz_option == 'Crime Incidents Heatmap':
            # Crime Incidents Heatmap
            st.subheader('Crime Incidents Heatmap')
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=data, x='longitude', y='latitude', fill=True, cmap='viridis', thresh=0, levels=100)
            plt.title('Heatmap of Crime Incidents')
            st.pyplot(plt)

        elif viz_option == 'Count Incidents by Severity':
            # Count Incidents by Severity
            st.subheader('Count Incidents by Severity')
            severity_counts = data['severity'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(y=severity_counts.index, x=severity_counts.values, palette='viridis')
            plt.title('Count of Incidents by Severity')
            plt.xlabel('Number of Incidents')
            plt.ylabel('Severity')
            st.pyplot(plt)

        elif viz_option == 'Incidents by Year and Month':
            # Incidents by Year and Month
            st.subheader('Incidents by Year and Month')
            data['year_month'] = data['incident_date'].dt.to_period('M')
            year_month_counts = data['year_month'].value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            year_month_counts.plot(kind='bar')
            plt.title('Incidents by Year and Month')
            plt.xlabel('Year-Month')
            plt.ylabel('Number of Incidents')
            st.pyplot(plt)

        elif viz_option == 'Elbow Method for Optimal k':
            # Elbow Method for Optimal k in KMeans
            st.subheader('Elbow Method for Optimal k')
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
            # Clustering
            st.subheader('K-Means Clustering')
            kmeans = KMeans(n_clusters=4)
            data['cluster'] = kmeans.fit_predict(coordinates)
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=data, palette='viridis')
            plt.title('Crime Clusters in Buffalo')
            st.pyplot(plt)

        elif viz_option == 'Top 10 Locations':
            # Top 10 Locations with Highest Crime
            st.subheader('Top 10 Locations with Highest Crime')
            top10_locations = data['address'].value_counts().head(10)
            st.bar_chart(top10_locations)

        # Model Selection
        st.subheader('Model Selection')
        model_option = st.selectbox('Select a Model', ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM', 'XGBoost'])

        if model_option == 'Logistic Regression':
            logistic_regression(data)
        elif model_option == 'Naive Bayes':
            naive_bayes(data)
        elif model_option == 'Random Forest':
            random_forest(data)
        elif model_option == 'SVM':
            svm(data)
        elif model_option == 'XGBoost':
            xgboost(data)



# Logistic Regression Function
def logistic_regression(data):
    st.subheader('Logistic Regression')
    X, y = preprocess_data(data)  # Preprocess the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Logistic Regression Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display_confusion_matrix(cm, y_test, "Logistic Regression Confusion Matrix")

# Naive Bayes Function
def naive_bayes(data):
    st.subheader('Naive Bayes')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Naive Bayes Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display_confusion_matrix(cm, y_test, "Naive Bayes Confusion Matrix")

# Random Forest Function
def random_forest(data):
    st.subheader('Random Forest')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Random Forest Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = model.feature_importances_
    plot_feature_importance(feature_importance, X.columns)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display_confusion_matrix(cm, y_test, "Random Forest Confusion Matrix")

# SVM Function
def svm(data):
    st.subheader('Support Vector Machine (SVM)')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardizing the data for SVM
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"SVM Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display_confusion_matrix(cm, y_test, "SVM Confusion Matrix")

# XGBoost Function
def xgboost(data):
    st.subheader('XGBoost')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"XGBoost Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = model.feature_importances_
    plot_feature_importance(feature_importance, X.columns)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display_confusion_matrix(cm, y_test, "XGBoost Confusion Matrix")

# Helper Function: Data Preprocessing
def preprocess_data(data):
    features = ['hour_of_day', 'latitude', 'longitude', 'month']
    X = data[features]
    y = LabelEncoder().fit_transform(data['severity'])
    return X, y

# Helper Function: Confusion Matrix Visualization
def display_confusion_matrix(cm, y_test, title):
    unique_labels = np.unique(y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    st.pyplot(plt)

# Helper Function: Feature Importance Plot
def plot_feature_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
