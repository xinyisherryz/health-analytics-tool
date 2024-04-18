import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Define the order of categories for the plots
age_order = ['18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 or older']
education_order = [
    'Less than high school', 'High school graduate',
    'Some college or technical school', 'College graduate'
]
income_order = [
    'Less than $15,000', '$15,000 - $24,999', '$25,000 - $34,999',
    '$35,000 - $49,999', '$50,000 - $74,999', '$75,000 or greater'
]
race_order = [
    'Non-Hispanic Black', 'Hispanic', 'Non-Hispanic White', 'Asian', 
    'Hawaiian/Pacific Islander', 'American Indian/Alaska Native', 'Other'
]

def load_model():
    return load('./model/gradient_boosting_model.joblib')

# Load the imputed CDC data
@st.cache_data
def load_imputed_data():
    return pd.read_csv('data/imputed_df.csv')

# User input features
def user_input_features():
    with st.sidebar:
        HighBP = st.number_input('High Blood Pressure (0: No, 1: Yes)', value=0)
        HighChol = st.number_input('High Cholesterol (0: No, 1: Yes)', value=0)
        CholCheck = st.number_input('Cholesterol Check in Last 5 Years (0: No, 1: Yes)', value=1)
        BMI = st.number_input('Body Mass Index', value=25)
        Stroke = st.number_input('History of Stroke (0: No, 1: Yes)', value=0)
        HeartDiseaseorAttack = st.number_input('Heart Disease or Attack (0: No, 1: Yes)', value=0)
        PhysActivity = st.number_input('Physical Activity in Last 30 Days (0: No, 1: Yes)', value=0)
        Fruits = st.number_input('Fruit Consumption Daily (0: No, 1: Yes)', value=1)
        Veggies = st.number_input('Vegetable Consumption Daily (0: No, 1: Yes)', value=1)
        HvyAlcoholConsump = st.number_input('Heavy Alcohol Consumption (0: No, 1: Yes)', value=0)
        AnyHealthcare = st.number_input('Access to Healthcare (0: No, 1: Yes)', value=1)
        GenHlth = st.number_input('General Health (scaled)', value=3)
        PhysHlth = st.number_input('Physical Health (scaled)', value=3)
        DiffWalk = st.number_input('Difficulty Walking (0: No, 1: Yes)', value=0)
        Sex = st.number_input('Sex (0: Female, 1: Male)', value=0)
        Age = st.number_input('Age', min_value=18, max_value=100, value=30)
        Education = st.number_input('Education Level', min_value=1, max_value=4, value=2)
        Income = st.number_input('Income Level', min_value=1, max_value=6, value=3)

    features = {
        'HighBP': HighBP, 'HighChol': HighChol, 'CholCheck': CholCheck,
        'BMI': BMI, 'Stroke': Stroke, 'HeartDiseaseorAttack': HeartDiseaseorAttack,
        'PhysActivity': PhysActivity, 'Fruits': Fruits, 'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump, 'AnyHealthcare': AnyHealthcare,
        'GenHlth': GenHlth, 'PhysHlth': PhysHlth, 'DiffWalk': DiffWalk,
        'Sex': Sex, 'Age': Age, 'Education': Education, 'Income': Income
    }
    return pd.DataFrame([features])

# Plot health data by category, e.g., age, education level, income, race/ethnicity group
def plot_health_data_by_category(df, category, health_conditions, demographic, user_selection):
    st.subheader(f"Health Conditions by {category}")

    # Set the order of X-axis based on the category
    if category == 'Age (years)':
        order = age_order
    elif category == 'Education':
        order = education_order
    elif category == 'Income':
        order = income_order
    elif category == 'Race/Ethnicity':
        order = race_order

    # Ask user to select a health risk factor
    condition = st.selectbox(f'Select the health risk factor:', health_conditions, key=f'condition_{category}')

    # Filter the data based on the selected category and health condition
    condition_data = df[(df['category'] == category) & (df['question'] == condition) & (~df[demographic].isin(['Data not reported']))]
    condition_data[demographic] = pd.Categorical(condition_data[demographic], categories=order, ordered=True)
    condition_data.sort_values(by=demographic, inplace=True)

    # Group the data by the demographic and calculate the average value
    grouped_data = condition_data.groupby(demographic)['value'].mean().dropna()

    # Check if the data is empty
    if grouped_data.empty:
        st.warning(f"No data available for {condition} within {category}.")
        return

    # Display the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    if category != 'Race/Ethnicity':
        sns.lineplot(x=grouped_data.index, y=grouped_data.values, marker='o', markersize=10, ax=ax) # Increase markersize
        if user_selection[category]:
            highlight_data = condition_data[condition_data[demographic] == user_selection[category]]
            sns.lineplot(x=highlight_data[demographic], y=highlight_data['value'], color='red', marker='o', markersize=14, ax=ax)  # Big red dots
    else:
        sns.barplot(x='value', y=demographic, data=condition_data, ci=None, ax=ax, palette=['#e0e0e0'] * len(order))  # Lighter colors for other bars
        if user_selection[category]:
            highlight_data = condition_data[condition_data[demographic] == user_selection[category]]
            sns.barplot(x='value', y=demographic, data=highlight_data, color='red', ax=ax)  # Highlight in red

    ax.set_title(f'Health Conditions by {category}')
    ax.set_xlabel(demographic.replace('_', ' '))
    ax.set_ylabel('Average Value')
    plt.xticks(rotation=45) if category != 'Race/Ethnicity' else plt.yticks(rotation=45)
    ax.grid(True)
    st.pyplot(fig)

# Main function
def main():
    st.title('Health Analytics Tool')

    # Load model and data
    model = load_model()
    df = load_imputed_data()

    # Create tabs for Diabetes Prediction and Data Exploration
    tab1, tab2 = st.tabs(["Diabetes Prediction", "More Exploration"])

    with tab1:
        # User inputs features and gets a prediction
        input_df = user_input_features()
        if st.button('Predict'):
            prediction = model.predict(input_df)
            st.write(f'Prediction: {"Diabetes" if prediction[0] == 1 else "No Diabetes"}')

    with tab2:
        st.header("Data Exploration")

        # Mapping of risk factors to full descriptions
        questions_mapping = {
            'Obesity': 'Percent of adults aged 18 years and older who have obesity',
            'No_Physical_Activity': 'Percent of adults who engage in no leisure-time physical activity',
            'Physical_Activity_150_2': 'Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic physical activity and engage in muscle-strengthening activities on 2 or more days a week',
            'Physical_Activity_150': 'Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic activity (or an equivalent combination)',
            'Muscle_Strengthening': 'Percent of adults who engage in muscle-strengthening activities on 2 or more days a week',
            'Physical_Activity_300': 'Percent of adults who achieve at least 300 minutes a week of moderate-intensity aerobic physical activity or 150 minutes a week of vigorous-intensity aerobic activity (or an equivalent combination)',
            'Fruit_Intake': 'Percent of adults who report consuming fruit less than one time daily',
            'Vegetable_Intake': 'Percent of adults who report consuming vegetables less than one time daily'
        }

        # Display the mapping as a table
        st.table(pd.DataFrame(list(questions_mapping.items()), columns=['Risk Factor', 'Description']))

        # User inputs personal information
        user_selection = {
            'Age (years)': st.sidebar.selectbox('Select your age group:', age_order),
            'Education': st.sidebar.selectbox('Select your education level:', education_order),
            'Income': st.sidebar.selectbox('Select your income level:', income_order),
            'Race/Ethnicity': st.sidebar.selectbox('Select your race/ethnicity:', race_order)
        }

        # Plot health conditions by category
        questions = list(questions_mapping.keys())
        for category in ['Age (years)', 'Education', 'Income', 'Race/Ethnicity']:
            plot_health_data_by_category(df, category, questions, 'demographic', user_selection)

if __name__ == '__main__':
    main()
