import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import pandas as pd
from PIL import Image


def streamlit_menu():
   
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  
                options=["Home", "Prediction", "SHAP","LIME and ELI5"],  
                icons=["house","activity", "x-square-fill","x-square-fill"],  
                menu_icon="cast", 
                default_index=0,  
            )
        return selected

    


selected = streamlit_menu()

def model_prediction():
    st.markdown("<h3 style='color:red'>Model Prediction</h3>", unsafe_allow_html=True)

    age=st.slider("Enter Your Age",17,90)

    workclass_list=['Federal-gov','Local-gov','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay']

    out1=st.selectbox("Select Your WorkClass",workclass_list)

    if(out1=='Federal-gov'):
        workclass=0
    elif(out1=='Local-gov'):
        workclass=1
    elif(out1=='Private'):
        workclass=2
    elif(out1=='Self-emp-inc'):
        workclass=3
    elif(out1=='Self-emp-not-inc'):
        workclass=4
    elif(out1=='State-gov'):
        workclass=5
    else:
        workclass=6


    fnlwgt=st.slider("Enter Your Financial Weight",13769,1484705)

    education_num_list=['Preschool','1st to 4th','5th to 6th','7th to 8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate']

    out2=st.selectbox("Select Your Educational Qualification",education_num_list)

    if(out2=='Preschool'):
        education_num=1
    elif(out2=='1st to 4th'):
        education_num=2
    elif(out2=='5th to 6th'):
        education_num=3
    elif(out2=='7th to 8th'):
        education_num=4
    elif(out2=='9th'):
        education_num=5
    elif(out2=='10th'):
        education_num=6
    elif(out2=='11th'):
        education_num=7
    elif(out2=='12th'):
        education_num=8
    elif(out2=='HS-grad'):
        education_num=9
    elif(out2=='Some-college'):
        education_num=10
    elif(out2=='Assoc-voc'):
        education_num=11
    elif(out2=='Assoc-acdm'):
        education_num=12
    elif(out2=='Bachelors'):
        education_num=13
    elif(out2=='Masters'):
        education_num=14
    elif(out2=='Prof-school'):
        education_num=15
    else:
        education_num=16


    marital_status_list=['Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Never-married','Separated','Widowed']

    out3=st.selectbox("Select Your marital status",marital_status_list)

    if(out3=='Divorced'):
        marital_status=0
    elif(out3=='Married-AF-spouse'):
        marital_status=1
    elif(out3=='Married-civ-spouse'):
        marital_status=2
    elif(out3=='Married-spouse-absent'):
        marital_status=3
    elif(out3=='Never-married'):
        marital_status=4
    elif(out3=='Separated'):
        marital_status=5
    else:
        marital_status=6

    occupation_list=['Adm-clerical','Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing','Handlers-cleaners','Machine-op-inspct','Other-service','Priv-house-serv','Prof-specialty','Protective-serv','Sales','Tech-support','Transport-moving']

    out4=st.selectbox("Select Your occupation",occupation_list)

    if(out4=='Adm-clerical'):
        occupation=0
    elif(out4=='Armed-Forces'):
        occupation=1
    elif(out4=='Craft-repair'):
        occupation=2
    elif(out4=='Exec-managerial'):
        occupation=3
    elif(out4=='Farming-fishing'):
        occupation=4
    elif(out4=='Handlers-cleaners'):
        occupation=5
    elif(out4=='Machine-op-inspct'):
        occupation=6
    elif(out4=='Other-service'):
        occupation=7
    elif(out4=='Priv-house-serv'):
        occupation=8
    elif(out4=='Prof-specialty'):
        occupation=9
    elif(out4=='Protective-serv'):
        occupation=10
    elif(out4=='Sales'):
        occupation=11
    elif(out4=='Tech-support'):
        occupation=12
    else:
        occupation=13

    relationship_list=['Husband','Not-in-family','Other-relative','Own-child','Unmarried','Wife']

    out5=st.selectbox("Select Your relationship",relationship_list)

    if(out5=='Husband'):
        relationship=0
    elif(out5=='Not-in-family'):
        relationship=1
    elif(out5=='Other-relative'):
        relationship=2
    elif(out5=='Own-child'):
        relationship=3
    elif(out5=='Unmarried'):
        relationship=4
    else:
        relationship=5

    race_list=['Amer-Indian-Eskimo','Asian-Pac-Islander','Black','Other','White']

    out6 = st.selectbox("Select Your race",race_list)

    if(out6=='Amer-Indian-Eskimo'):
        race=0
    elif(out6=='Asian-Pac-Islander'):
        race=1
    elif(out6=='Black'):
        race=2
    elif(out6=='Other'):
        race=3
    else:
        race=4

    options = ['Male','Female']

    out7= st.selectbox('Select Your Gender:', options)

    if(out7=='Male'):
        sex=1
    else:
        sex=0

    hours_per_week=st.slider("Enter Your work hours per week",1,99)

    countries=['United-States', 'Mexico', 'Greece', 'Vietnam', 'China', 'Taiwan',
        'India', 'Philippines', 'Trinadad&Tobago', 'Canada', 'South',
        'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran', 'England',
        'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba', 'Ireland',
        'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic', 'Haiti',
        'Hungary', 'Columbia', 'Guatemala', 'El-Salvador', 'Jamaica',
        'Ecuador', 'France', 'Yugoslavia', 'Portugal', 'Laos', 'Thailand',
        'Outlying-US(Guam-USVI-etc)', 'Scotland']

    out8=st.selectbox("Select Your Native Country",countries)

    if (out8=='United-States'):
        native_country=38
    elif (out8=='Mexico'):
        native_country=25
    elif (out8=='Greece'):
        native_country=11
    elif (out8=='Vietnam'):
        native_country=39
    elif (out8=='China'):
        native_country=2
    elif (out8=='Taiwan'):
        native_country=35
    elif (out8=='India'):
        native_country=18
    elif (out8=='Philippines'):
        native_country=29
    elif (out8=='Trinadad&Tobago'):
        native_country=37
    elif (out8=='Canada'):
        native_country=1
    elif (out8=='South'):
        native_country=34
    elif (out8=='Holand-Netherlands'):
        native_country=14
    elif (out8=='Puerto-Rico'):
        native_country=32
    elif (out8=='Poland'):
        native_country=30
    elif (out8=='Iran'):
        native_country=19
    elif (out8=='England'):
        native_country=8
    elif (out8=='Germany'):
        native_country=10
    elif (out8=='Italy'):
        native_country=21
    elif (out8=='Japan'):
        native_country=23
    elif (out8=='Hong'):
        native_country=16
    elif (out8=='Honduras'):
        native_country=15
    elif (out8=='Cuba'):
        native_country=4
    elif (out8=='Ireland'):
        native_country=20
    elif (out8=='Cambodia'):
        native_country=0
    elif (out8=='Peru'):
        native_country=28
    elif (out8=='Nicaragua'):
        native_country=26
    elif (out8=='Dominican-Republic'):
        native_country=5
    elif (out8=='Haiti'):
        native_country=13
    elif (out8=='Hungary'):
        native_country=17
    elif (out8=='Columbia'):
        native_country=3
    elif (out8=='Guatemala'):
        native_country=12
    elif (out8=='El-Salvador'):
        native_country=7
    elif (out8=='Jamaica'):
        native_country=22
    elif (out8=='Ecuador'):
        native_country=6
    elif (out8=='France'):
        native_country=9
    elif (out8=='Yugoslavia'):
        native_country=40
    elif (out8=='Portugal'):
        native_country=31
    elif (out8=='Laos'):
        native_country=24
    elif (out8=='Thailand'):
        native_country=36
    elif (out8=='Outlying-US(Guam-USVI-etc)'):
        native_country=27
    else:
        native_country=33

   
    def predict(): 
        model = joblib.load('adult.joblib')
        row = np.array([age,workclass,fnlwgt,education_num,marital_status,occupation,relationship,race,sex,hours_per_week,native_country]) 
        X = pd.DataFrame([row], columns = ['age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex', 'hours.per.week',
        'native.country'])
        prediction = model.predict(X)
        if prediction[0] == 1: 
          st.success('Income Greater Than $50k')
        else: 
           st.error('Income Less than or Equal to $50k')
    trigger = st.button('Predict', on_click=predict)



if selected == "Home":
   st.title('Explainable AI')

   st.markdown("<h5>Aravindambalavanan J(2022179044)</h5>",unsafe_allow_html=True)
   st.markdown("<h5>Sanjayprakash M(2022179043)</h5>",unsafe_allow_html=True)
   st.markdown("<h5>Vinoth Kumar A(2022179041)</h5>",unsafe_allow_html=True)

   st.markdown("Explanation and Demo(https://drive.google.com/file/d/1fq7flJrXTvOmNLJ-qGSlHUp5gizT7M6e/view?usp=sharing)")

   st.write("Dataset Used - UCI Adult Census")

   st.markdown("Model(https://github.com/Aravindambalavanan/XAI).")

   st.write("Libraries used for Explainablity and Model Interpretability-SHAP,LIME,ELI5")

   st.markdown("""Explainable AI (XAI), also known as Interpretable AI or Explainable Machine Learning, aims to make the inner workings of AI models more understandable to humans. It focuses on two key goals:

1. Understanding model decisions:

This involves gaining insights into how AI models arrive at their predictions or conclusions.
XAI techniques help explain which features contribute most to a specific prediction, what factors influence the model's decision-making process, and how different scenarios might affect the outcome.
               
2. Building trust in AI systems:

By understanding how AI models work, users can gain trust in their reliability and accuracy.
This is particularly important in high-stakes applications where transparency and accountability are crucial.
               
Here are some key benefits of XAI:

Improved debugging and model analysis:
By understanding how models work, developers can identify and address potential biases, errors, and performance issues.
               
Enhanced decision-making: 
I helps users interpret model outputs and make more informed decisions based on their insights.
               
Increased user acceptance and trust: 
When users understand how AI models work, they are more likely to accept and trust their recommendations and predictions.
               
Compliance with regulations:
In some industries, such as finance and healthcare, regulations require AI models to be explainable and transparent.""")
if selected == "Prediction":
     model_prediction()
if selected == "SHAP":
    st.title("SHAP")

    st.markdown("""
    SHAP (SHapley Additive exPlanations) is a library in Python that helps explain the output of machine learning models. It is based on Shapley values, a concept from cooperative game theory, to assign a value to each feature in a prediction. Shapley values provide a fair way of distributing a value among a group of contributors.

    In the context of machine learning, SHAP values aim to attribute the prediction of a model to each individual feature, helping to understand the impact of each feature on the model's output for a particular instance. This can be useful for model interpretation, debugging, and gaining insights into the factors that influence model predictions.
        
    The SHAP library provides tools for various types of models, including tree-based models, linear models, and black-box models. It can be used to generate summary plots, force plots, and other visualizations to help users interpret the output of their machine learning models.""")


    summary_bar_plot = Image.open("plots/summary_bar_plot.png")

    st.markdown("<h3>Summary bar Plot</h3>", unsafe_allow_html=True)

    st.image(summary_bar_plot, caption='SHAP Summary bar Plot', use_column_width=True)

    st.markdown(""" 
    The most important feature for predicting whether someone's income is <=50k or >50k is relationship. 
                
    People who are in a relationship (relationship = Husband, Wife, or Not-in-family) are more likely to earn <=50k per year.

    Other important features include education.num, fnlwgt, workclass, age, and occupation. People with less education, lower fnlwgt values, lower-paying occupations, and who are younger are more likely to earn <=50k per year.

    The features sex, race, marital.status, and native.country have less of an impact on income, but they can still be influential in some cases. For example, people who are female (sex = Female) or who are married (marital.status = Married-civ-spouse) are more likely to earn <=50k per year.

    The feature hours.per.week has the least impact on income.

    Here is a more detailed explanation of each feature:

    relationship: People who are in a relationship (relationship = Husband, Wife, or Not-in-family) are more likely to have children and other financial responsibilities, which can make it difficult for them to earn a high income.

    education.num: People with less education (education.num < 12) have fewer skills and knowledge, which makes them less valuable to employers.

    fnlwgt: Fnlwgt is a weight factor that is used to adjust for the fact that the training data is not perfectly representative of the population. People with lower fnlwgt values are more likely to be from underrepresented groups, such as minorities and immigrants.

    workclass: Workclass is a categorical variable that indicates the type of job that a person has. Some workclasses, such as Private, are associated with lower incomes than others.

    age: Younger people (age < 40) are more likely to be early in their careers, which means that they have not had as much time to accumulate experience and earn a high income.

    occupation: Occupation is a strong predictor of income. Certain occupations, such as Service and Handlers-cleaners, pay significantly less than others.

    sex: Women (sex = Female) tend to earn less than men (sex = Male) on average, even after controlling for other factors such as education and occupation. This is due to a number of factors, including gender discrimination and the fact that women are more likely to take time out of the workforce for childcare and other family responsibilities.

    race: People of color (race = Black, Hispanic, Asian-Pac-Islander, Other) tend to earn less than white people (race = White) on average, even after controlling for other factors such as education and occupation. This is due to a number of factors, including racial discrimination and the fact that people of color are more likely to live in high-poverty areas with fewer job opportunities.

    marital.status: Married people (marital.status = Married-civ-spouse) tend to earn less than unmarried people (marital.status = Divorced, Married-sep, Never-married, Widowed) on average, even after controlling for other factors such as education and occupation. This is likely due to the fact that married people are more likely to have children and other financial responsibilities.

    native.country: Native country (native.country = United-States) has a relatively small impact on income, even after controlling for other factors such as education and occupation. However, people who are not from the United States (native.country != United-States) are slightly more likely to earn <=50k per year.""")

    summary_plot = Image.open("plots/summary_plot.png")

    st.markdown("<h3>Summary Plot(For Single Instance)</h3>", unsafe_allow_html=True)

    st.image(summary_plot, caption='SHAP Summary Plot', use_column_width=True)

    st.markdown("""
    From the above plot we can understand.
                
    The most important feature for predicting income is education.num, which is the number of years of education. People with more education are more likely to earn more than 50k per year.

    Other important features include age, hours.per.week, occupation, and fnlwgt. People who are older, work more hours per week, have higher-paying occupations, and have higher fnlwgt values are more likely to earn more than 50k per year.

    The features workclass, sex, race, and marital.status have less of an impact on income, but they can still be influential in some cases. For example, people who are self-employed (workclass = Self-emp-not-inc) are more likely to earn more than 50k per year, while people who are married (marital.status = Married-civ-spouse) are also more likely to earn more than 50k per year.
    The native.country feature has the least impact on income.

    Here is a more detailed explanation of each feature:

    education.num: The number of years of education is a strong predictor of income. This is because people with more education have more skills and knowledge, which makes them more valuable to employers.
                
    age: Age is also a predictor of income, but the relationship is more complex. In general, people's earnings increase with age until they reach their mid-career, and then they start to decline. However, there are many exceptions to this rule. For example, people who work in certain professions, such as doctors and lawyers, may continue to earn more money as they get older.
                
    hours.per.week: People who work more hours per week are more likely to earn more money. This is because they have more time to earn money. However, it is important to note that there is a point at which working more hours per week starts to have a negative impact on productivity.
                
    occupation: Occupation is a strong predictor of income. Certain occupations, such as doctors and lawyers, pay significantly more than others.
                
    fnlwgt: Fnlwgt is a weight factor that is used to adjust for the fact that the training data is not perfectly representative of the population.
                
    workclass: Workclass is a categorical variable that indicates the type of job that a person has. Some workclasses, such as Self-emp-not-inc, are associated with higher incomes than others.
                
    sex: Sex is a categorical variable that indicates whether a person is male or female. There is a small gender pay gap in the United States, with men earning slightly more than women on average.
                
    race: Race is a categorical variable that indicates the race of a person. There is a racial income gap in the United States, with white people earning more than people of other races on average.
                
    marital.status: Marital status is a categorical variable that indicates the marital status of a person. Married people tend to earn more than unmarried people, on average.
                
    native.country: Native country is a categorical variable that indicates the country of birth of a person. There are some income differences between people from different countries, but these differences are relatively small compared to the differences between people with different levels of education, occupation, and other factors.
                
    It is important to note that the SHAP plot shows the average impact of each feature on income. This means that there will be some people who deviate from this average. For example, some people with less education may still earn more than $50k per year, if they have other factors that work in their favor, such as a high-paying occupation or a lot of experience.""")

    st.markdown("<h3>Dependence Plots(For Most Important Features)</h3>", unsafe_allow_html=True)

    image_paths = ['plots/relationship_dependence_plot.png', 'plots/education_num_dependence_plot.png', 'plots/age_dependence_plot.png', 'plots/hours_per_week_dependence_plot.png']

    def create_image_collage(image_paths, collage_size=(500, 500), image_size=(250, 250)):
        collage = Image.new('RGB', collage_size, (255, 255, 255))
        
        for i, image_path in enumerate(image_paths):
            img = Image.open(image_path).resize(image_size)
            collage.paste(img, ((i % 2) * image_size[0], (i // 2) * image_size[1]))
        
        return collage

    collage = create_image_collage(image_paths)

    st.image(collage, caption='SHAP Dependence Plots', use_column_width=True)

    st.markdown("<h5 style='color:red'>Relationship and Age Dependencey</h5>", unsafe_allow_html=True)

    st.markdown(""" 
    The SHAP dependence plot with relationship and age shows that the relationship has a larger impact on the SHAP values for younger people than for older people. This means that the difference in SHAP values between people in a relationship and people not in a relationship is greater for younger people.

    One possible explanation for this is that younger people are more likely to be early in their careers and in their relationships. This means that their relationship status may have a greater impact on their financial responsibilities and opportunities. For example, younger people in a relationship may be more likely to have children, which can increase their childcare expenses and make it more difficult for them to work long hours.

    Another possible explanation is that there is a generational shift in attitudes towards relationships. Younger people may be more likely to view relationships as a partnership, which can lead to a more equitable division of financial responsibilities. This could explain why the difference in SHAP values between people in a relationship and people not in a relationship is smaller for older people.

    Overall, the SHAP dependence plot provides a useful way to understand the complex relationship between relationship, age, and other factors that may impact income.

    Here are some specific observations from the plot:

    Young people in a relationship have a higher expected income than young people not in a relationship. This difference is greater for younger people than for older people.

    Older people in a relationship have a slightly lower expected income than older people not in a relationship. This difference is smaller for older people than for younger people.

    The relationship between relationship and expected income is non-linear. This means that the impact of relationship on expected income is not constant. The impact is greater for younger people and decreases as people get older.

    It is important to note that the SHAP dependence plot shows the average impact of relationship and age on expected income. This means that there will be some people who deviate from this average. For example, some young people not in a relationship may still earn more money than young people in a relationship.""")

    st.markdown("<h5 style='color:red'>Education and Relationship Dependency</h5>", unsafe_allow_html=True)

    st.markdown("""
    There is a negative interaction between relationship and education num. This means that the negative effect of being in a relationship on expected income is greater for people with more education.
    To put it another way, people in a relationship with more education are expected to earn less money than people who are unmarried with the same level of education.

    Here is an example to illustrate this:

    Person A is unmarried and has 16 years of education. Person A's expected income is 100,000 per year.
    Person B is in a relationship and has 16 years of education. Person B's expected income is 95,000 per year.
    Person C is unmarried and has 12 years of education. Person C's expected income is 70,000 per year.
    Person D is in a relationship and has 12 years of education. Person D's expected income is 65,000 per year.
    As you can see, Person B is expected to earn less money than Person A, even though they have the same level of education. This is because Person B is in a relationship, which has a negative effect on expected income.

    It is important to note that the SHAP dependence plot shows the average impact of education num and relationship on expected income. This means that there will be some people who deviate from this average. For example, some people in a relationship with more education may still earn more money than people who are unmarried with the same level of education.

    Overall, the SHAP dependence plot provides a useful way to understand the complex relationship between education num, relationship, and expected income.

    Possible explanations for the negative interaction between relationship and education num:

    People in a relationship may be more likely to choose lower-paying jobs that are more flexible or closer to home, in order to balance their work and family responsibilities.
    People in a relationship may be more likely to take time out of the workforce to raise children or care for elderly relatives. This can reduce their overall earnings and career advancement opportunities.
    People in a relationship may be more likely to have shared financial obligations, such as a mortgage or childcare expenses. This can make it more difficult for them to save money and invest in their own careers.""")

    st.markdown("<h5 style='color:red'>Age and Education Dependency</h5>", unsafe_allow_html=True)

    st.markdown("""
    There is a positive interaction between age and education num. This means that the positive effect of education num on expected income is greater for older people.
    To put it another way, older people with more education are expected to earn more money than older people with less education.

    Here is an example to illustrate this:

    Person A is 25 years old and has 12 years of education. Person A's expected income is 50,000 per year.
    Person B is 45 years old and has 12 years of education. Person B's expected income is 60,000 per year.
    Person C is 25 years old and has 16 years of education. Person C's expected income is 60,000 per year.
    Person D is 45 years old and has 16 years of education. Person D's expected income is 70,000 per year.
    As you can see, Person D is expected to earn more money than Person C, even though they have the same level of education. This is because Person D is older, and the positive effect of education num on expected income is greater for older people.

    It is important to note that the SHAP dependence plot shows the average impact of age and education num on expected income. This means that there will be some people who deviate from this average. For example, some younger people with more education may still earn more money than older people with less education.

    Overall, the SHAP dependence plot provides a useful way to understand the complex relationship between age, education num, and expected income.

    Possible explanations for the positive interaction between age and education num:

    Older people with more education have had more time to accumulate experience and build their careers. This makes them more valuable to employers and allows them to command higher salaries.
    Older people with more education are more likely to be in leadership positions, which typically come with higher salaries and benefits.
    Older people with more education are more likely to have invested in their own careers, such as through continuing education or professional development. This can lead to higher earnings and career advancement opportunities.""")

    st.markdown("<h5 style='color:red'>Education and Relationship Dependency</h5>", unsafe_allow_html=True)

    st.markdown("""
    There is a positive interaction between relationship and hours per week. This means that the positive effect of working more hours on expected income is greater for people in a relationship.
    To put it another way, people in a relationship who work more hours per week are expected to earn more money than people who are unmarried and work the same number of hours per week.

    Here is an example to illustrate this:

    Person A is unmarried and works 40 hours per week. Person A's expected income is 50,000 per year.
    Person B is in a relationship and works 40 hours per week. Person B's expected income is 55,000 per year.
    Person C is unmarried and works 60 hours per week. Person C's expected income is 60,000 per year.
    Person D is in a relationship and works 60 hours per week. Person D's expected income is 65,000 per year.
    As you can see, Person D is expected to earn more money than Person C, even though they both work the same number of hours per week. This is because Person D is in a relationship, which has a positive effect on expected income.

    It is important to note that the SHAP dependence plot shows the average impact of hours per week and relationship on expected income. This means that there will be some people who deviate from this average. For example, some unmarried people who work long hours may still earn more money than people in a relationship who work fewer hours.""")
if selected == "LIME and ELI5":
    st.title("LIME: Local Interpretable Model-Agnostic Explanations")

    st.markdown("""
    LIME, which stands for Local Interpretable Model-Agnostic Explanations, is a technique for explaining the predictions of any machine learning model. It works by approximating the model locally around a particular data point, and then using a simple, interpretable model to explain the prediction.

    Here are some of the key features of LIME:

    Model-agnostic: LIME can be used to explain the predictions of any machine learning model, regardless of its underlying architecture or algorithm. This makes it a very versatile tool for understanding how black-box models make decisions.

    Locally faithful: LIME explanations are faithful to the model's predictions in the local vicinity of the data point. This means that the explanation is not just a general description of the model, but is specific to how the model behaves for that particular data point.

    Interpretable: LIME explanations are generated using simple, interpretable models, such as linear regression or decision trees. This makes them easy to understand for non-experts in machine learning.

    Counterfactual: LIME can also be used to generate "counterfactuals," which are hypothetical data points that would have been classified differently by the model. This can help to identify the features that are most important for the model's prediction.""")

    st.markdown("<h5>local explanation for the prediction for a particular Datapoint</h5>", unsafe_allow_html=True)

    LIME_plot = Image.open("plots/LIME_plot.png")

    st.image(LIME_plot, use_column_width=True)

    st.markdown("""
    The data point in question is a person who is 38 years old, has 16 years of education, and has an income of $50,000 per year. 
    The model has predicted that this person is likely to have a college degree.

    The LIME plot shows the features that are most important for the model's prediction. 

    The features are ranked by their importance, with the most important feature at the top of the plot. The value of each feature is shown on the x-axis, and the importance of the feature is shown on the y-axis.

    The most important feature for the model's prediction is education.num. This is because people with more education are more likely to have a college degree. The next most important feature is age. This is because older people are more likely to have had time to complete a college education. The other features, such as workclass and occupation, have less of an impact on the model's prediction.

    The LIME plot also shows the counterfactual values for each feature. The counterfactual value is the value that would have been needed for the feature in order for the model to predict that the person does not have a college degree. For example, if the person had 12 years of education instead of 16 years of education, the model would have predicted that they do not have a college degree.""")

    st.title("ELI5")

    ELI5_plot = Image.open("plots/ELI5_plot.png")

    st.image(ELI5_plot, width=300)

    st.markdown("""
    The plot shows that the relationship feature has the biggest impact on the model's prediction. This is because people in a relationship are more likely to have children and other financial responsibilities, which can make it difficult for them to have a high income.

    The education.num feature also has a big impact on the model's prediction. This is because people with more education have more skills and knowledge, which makes them more valuable to employers.

    The other features, such as fnlwgt, workclass, age, and occupation, also have an impact on the model's prediction, but to a lesser extent.

    The plot also shows that the impact of each feature changes as the value of the feature changes. For example, the impact of relationship on the model's prediction is greater for people with less education.""")