import pandas as pd
import warnings
warnings.filterwarnings('ignore')




#renewals dataset

renewals = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2023/renewals.csv')

renewals.isna().sum()
print(renewals.nunique())
print(renewals.dtypes)
print(renewals.describe())

month_mapping = ({'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12})

renewals['purchase_month'] = renewals['purchase_month'].map(month_mapping)

renewals = pd.DataFrame(renewals)

renewals.isna().sum()

#coping renwewals as a new dataframe

df_renew = renewals.copy() 
df_renew = df_renew.drop(['customer_num','zip_code'],axis=1)





df_renew.isna().sum()

import statsmodels.api as sm


X = df_renew.drop('renewed',axis=1)
y = df_renew['renewed']

model_1 = sm.OLS(X,y).fit()
print(model_1.summary())



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier

rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
BC = BaggingClassifier()


from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve


def evaluate_model(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    pred_prob = model.predict_proba(X_test_scaled)[::,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    print(f'{model.__class__.__name__}, --Accuracy score-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob


lr_pred,lr_pred_prob = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test)
rfc_pred,rfc_pred_prob = evaluate_model(rfc, X_train_scaled, X_test_scaled, y_train, y_test)
gbc_pred,gbc_pred_prob = evaluate_model(gbc, X_train_scaled, X_test_scaled, y_train, y_test)
BC_pred,BC_pred_prob = evaluate_model(BC, X_train_scaled, X_test_scaled, y_train, y_test)


import matplotlib.pyplot as plt


def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    
    
ROC(y_test,lr_pred_prob,lr)
ROC(y_test,rfc_pred_prob,rfc)
ROC(y_test,gbc_pred_prob,gbc)
ROC(y_test,BC_pred_prob,BC)
plt.legend()
plt.show()


#import season passes dataset

season_passes = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2023/season_passes.csv')

season_passes.isna().sum()
season_passes.describe()


season_passes = season_passes.copy()
season_passes.nunique()
season_passes['purchase_month'] = season_passes['purchase_month'].map(month_mapping)



season_passes = pd.DataFrame(season_passes)

#creating a dictionary for pass level(categorical)

pass_level_mapping = ({'Gold':1,'Silver':2,'Bronze':3})

season_passes['pass_level'] = season_passes['pass_level'].map(pass_level_mapping)

season_passes = pd.DataFrame(season_passes)


df_season_passes = season_passes.copy()

df_season_passes.info()
df_season_passes = df_season_passes.drop('zip_code',axis=1)

df_season_passes.isna().sum()
df_season_passes.describe()
df_season_passes.drop_duplicates(inplace=True)


#merging season passes


import pandas as pd

# Assuming you have renewals and season_passes DataFrames

# Step 1: Merge renewals and season_passes
merged_df = pd.merge(renewals, season_passes, on='customer_num', how='inner')

# Step 2: Sort the merged DataFrame
merged_df = merged_df.sort_values(by=['purchase_year', 'renewed'])

# Step 3: Remove duplicates
merged_df.drop_duplicates(inplace=True)

# Step 4: Copy the DataFrame
df_merged = merged_df.copy()

# Step 5: Drop unnecessary columns
#columns_to_drop = ['zip_code_x', 'zip_code_y', 'purchase_month_x', 'purchase_month_y']
#df_merged = df_merged.drop(columns_to_drop, axis=1)

# Step 6: Remove duplicates again
df_merged.drop_duplicates(inplace=True)

# Step 7: Display summary statistics
print(df_merged.describe())

# Step 8: Add 'renewed_in_2022' column
df_merged['renewed_in_2022'] = (df_merged['purchase_year'] == 2022) & (~df_merged['pass_level'].isna())

# Step 9: Separate into renewals and non-renewals for 2022
renewals_2022 = df_merged[df_merged['renewed_in_2022']]
non_renewals_2022 = df_merged[~df_merged['renewed_in_2022']]

# Step 10: Drop customers who bought a season pass only in 2022
non_renewals_2022 = non_renewals_2022[~((non_renewals_2022['purchase_year'] == 2022) & ~non_renewals_2022['renewed_in_2022'])]

# Step 11: Drop unnecessary columns for renewals and non-renewals
columns_to_drop_2022 = ['zip_code_x', 'zip_code_y', 'purchase_month_x', 'purchase_month_y']
renewals_2022 = renewals_2022.drop(columns_to_drop_2022, axis=1)
non_renewals_2022 = non_renewals_2022.drop(columns_to_drop_2022, axis=1)

# Step 12: Remove duplicates for renewals and non-renewals
renewals_2022.drop_duplicates(inplace=True)
non_renewals_2022.drop_duplicates(inplace=True)

# Step 13: Display summary statistics for renewals and non-renewals in 2022
print("Renewals in 2022:")
print(renewals_2022.describe())

print("\nNon-Renewals in 2022:")
print(non_renewals_2022.describe())

# Step 14: Add columns indicating if the customer held annual passes in 2021 and 2022
df_merged['held_annual_pass_2021'] = (df_merged['purchase_year'] == 2021) & (~df_merged['pass_level'].isna())
df_merged['held_annual_pass_2022'] = (df_merged['purchase_year'] == 2022) & (~df_merged['pass_level'].isna())

# Step 15: Create a new column representing the overall condition
df_merged['held_annual_pass_both_years'] = (df_merged['held_annual_pass_2021'] & df_merged['held_annual_pass_2022']).astype(int)

# Step 16: Drop unnecessary columns
df_annual_pass_both_years = df_merged[['customer_num', 'held_annual_pass_both_years']].copy()

# Step 17: Display the resulting DataFrame
print(df_annual_pass_both_years.head())

df_merged.info()
