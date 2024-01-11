# Streamlit
import streamlit as st
import pandas as pd
import attrition_model as am

def fn_st(model,df,pp):
    st.header("mlop project")
    # upload_bn = st.button("Bulk upload")

    csv_file = st.file_uploader("Bulk upload")
    if csv_file:
        df_test = pd.read_csv(csv_file)
        clicked = st.button("view file")
        if clicked:
            X = df_test.drop(['Attrition'], axis=1)
            st.dataframe(X)
        else:
            st.write("No file uploaded yet")
        show_result = st.button("click to view result")
        if show_result:
            # am.fn_pipe(df,'train',model,0)
            st.dataframe(am.fn_pipe(df_test,'test',model,pp))
    
    # detail_bn = st.button("Enter details")

    age = st.number_input("enter the age")
    dept = st.selectbox("Choose a department",('Sales','Human Resources','Research & Development'))
    dist = st.number_input("Distance from home")
    edu = st.select_slider("education",[1,2,3,4,5])
    edu_field = st.selectbox("Choose a Education field",('Human Resources','Life Sciences','Marketing','Medical','Other','Technical Degree'))
    env = st.select_slider("Choose environment satisfaction",options=[1,2,3,4])
    job = st.select_slider("Choose job satisfaction",options=[1,2,3,4])   
    marital = st.radio("maritial status",['Single','Married','Divorced'])
    income = st.number_input("Monthly salary") 
    comp = st.select_slider("no: of companies worked",[item for item in range(0,10)])
    wl = st.select_slider("no: of companies worked",[1,2,3,4])
    years = st.number_input("Years at company")
    result = st.button("Predict")
    if result:
        df = pd.DataFrame({'Age':age,
                        'Department':dept,
                        'DistanceFromHome':dist,
                        'Education':edu,
                        'EducationField':edu_field,
                        'EnvironmentSatisfaction':env,
                        'JobSatisfaction':job,
                        'MaritalStatus':marital,
                        'MonthlyIncome':income,
                        'NumCompaniesWorked':comp,
                        'WorkLifeBalance':wl,
                        'YearsAtCompany':years,
                        'Attrition':'No'                          
                        },index=[0])
        X = am.fn_pipe(df,'test',model,pp)
        st.dataframe(X)
    # eval_bn = st.button("eval diff models")
    # if eval_bn:
    # options = st.multiselect('Choose algo',['Logistic', 'RF', 'Adaboost', 'GBC', 'LGBM', 'XGB'])
    # st.write(options)
    # return (options)
    

    
    