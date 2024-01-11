import attrition_model as am
import attrition_mlflow as al
import attrition_streamlit as ast
import os
import pandas as pd


# os.chdir(r"")
df = pd.read_csv("IBM.csv")

model,pp,X_train,X_test,y_train,y_test,model_list = am.fn_model(df)
# print(model)
al.fn_mlflow(model,X_train,X_test,y_train,y_test,model_list)
ast.fn_st(model,df,pp)



