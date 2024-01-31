import pandas as pd
import warnings
warnings.filterwarnings('ignore')

file_path = "C:/Users/katsa/Downloads/cps_00103.csv.gz"
data = pd.read_csv(file_path, compression='gzip')

data.columns
sampling_fraction = 0.2
random_state = 42

sampled_df = data.sample(frac=sampling_fraction, random_state=random_state)

features = ["YEAR", "SEX", "AGE", "RACE", "HISPAN","PAIDHOUR","EARNWEEK","HOURWAGE","UHRSWORK1","EDUC"]
weight_variable = "EARNWT"


df = sampled_df[features + [weight_variable]]

sampled_df.head(10)
df.head(10)
df.isna().sum()
df = df[df['YEAR'] == 2021]

df = df.drop('EARNWT',axis=1)

valid_conditions = (df['EARNWEEK'] > 0) & (df['EARNWEEK'] < 9999.99) & (df['UHRSWORK1'] > 0) & (df['UHRSWORK1'] < 997)
hourly_wage_condition = (df['PAIDHOUR'] == 2) & (df['HOURWAGE'] > 0) & (df['HOURWAGE'] < 99)


df.loc[valid_conditions & hourly_wage_condition, 'HOURWAGE'] = df['HOURWAGE']


not_paid_hour_condition = (df['PAIDHOUR'] == 1)
df.loc[valid_conditions & not_paid_hour_condition, 'HOURWAGE'] = df['EARNWEEK'] / df['UHRSWORK1']

df['HOURWAGE'] = df['HOURWAGE'].clip(lower=0, upper=997)



df['Low_Wage_Worker'] = [1 if x <= 15 else 0 for x in df['HOURWAGE']]

df.isna().sum()

df['YEAR'].describe()

df['RACE'].nunique()
#race



race_ = {100: 'white', 200: 'Black', 651: 'Asian'}
df['RACE'] = df['RACE'].apply(lambda x: race_[x] if x in race_ and x > 0 else 'other')



df['EDUC'].describe()

less_than_high_school = (df['EDUC'] <= 60)
high_school = (df['EDUC'] >= 70) & (df['EDUC'] <= 73)
some_college = (df['EDUC'] >= 80) & (df['EDUC'] <= 100)
college = (df['EDUC'] >= 110)


df.loc[less_than_high_school, 'EDUC'] = 'Less than High School'
df.loc[high_school, 'EDUC'] = 'High School'
df.loc[some_college, 'EDUC'] = 'Some College'
df.loc[college, 'EDUC'] = 'College'


df.info()
df.isna().sum()
print(df.dtypes)


education_mapping = {'Less than High School':0,'High School':1,'Some College':2,'College':3}
df['EDUC'] = df['EDUC'].map(education_mapping)

race_mapping = {'white':0,'Black':1,'Asian':2,'other':3}
df['RACE'] = df['RACE'].map(race_mapping)
df['RACE'].nunique()
low_wage_df = df[df['Low_Wage_Worker'] == 1]

# male = 1, Female = 2
df['SEX'] = df['SEX'].map({1:0,2:1})




import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(), annot=True)
plt.show()


def subplots(df):
    _, axs = plt.subplots(2,2,figsize=(15,8))
    sns.barplot(x='EDUC',y='RACE',ax=axs[0,0],data=df)
    axs[0,0].set_xlabel('0 = less Than High School Education; 1 = High School; 2= Some College; 3= College')
    axs[0,0].set_title('Low Wage Worker by Education and Race')
    sns.barplot(x='SEX', y='Low_Wage_Worker',ax=axs[0,1],data=df)
    axs[0,1].set_title('Low Wage Worker by Sex')
    axs[0,1].set_xlabel('0 == Male; 1 == Femal')
    sns.boxplot(x='RACE',y='AGE',hue='Low_Wage_Worker', ax=axs[1,0],data=df)
    axs[1,0].set_title('Count of Low Wage Worker by Race')
    axs[1,0].set_xlabel('0 = White; 1 = Black; 2 = Asian; 3 = other race')
    sns.violinplot(x='YEAR',y='EDUC',hue='Low_Wage_Worker', ax=axs[1,1],data=df)
    axs[1,1].set_title('Count of Low Wage Worker by Year')
    plt.tight_layout()

subplots(df)


# race and age between low age workers
plt.figure(figsize=(12,5))
sns.stripplot(x='RACE',y='AGE',data=low_wage_df,hue='EDUC')
plt.title('Distribution of Low-Wage Workers at Each Age within Each Race Group')
plt.xlabel('Race (1 == white; 2 == black; 3 == Asian; 4 == other)')
plt.ylabel('Age')
plt.show()


#Age and education of low age workers
plt.figure(figsize=(12,5))
sns.barplot(x='EDUC',y='AGE',data=low_wage_df,hue='EDUC')
plt.title('Distribution of Low-Wage Workers at Each Age at each Education Level')
plt.xlabel('Education Level : 0 = No High School Degree; 1 = High School Degree; 2 == Some College; 3 == College Degree')
plt.ylabel('Age')
plt.legend(title='Education Level')
plt.show()




import statsmodels.api as sm

#model for race/ethic differences leading to low_wage workforce

X_race = sm.add_constant(df['RACE'])
y_wage = df['Low_Wage_Worker']
model_race = sm.OLS(exog=X_race,endog=y_wage).fit()
print(model_race.summary())



# full model. Dependent Variable is Weekly Earnings(EARNWEEK)
X_earn = df.drop('EARNWEEK',axis=1)
y_earn = df['EARNWEEK']

full_model = sm.OLS(exog=sm.add_constant(X_earn),endog=y_earn).fit()
print(full_model.summary())
print(f'coefficients for the full model: {full_model.params}')


from sklearn.model_selection import train_test_split

X_earn_train,X_earn_test,y_earn_train,y_earn_test = train_test_split(X_earn,y_earn,test_size=.20,random_state=42)


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X_earn_train_scaled = ms.fit_transform(X_earn_train)
X_earn_test_scaled = ms.transform(X_earn_test)


from sklearn.linear_model import LinearRegression,Ridge,Lasso
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor
rfr = RandomForestRegressor()
GBR = GradientBoostingRegressor()
BR = BaggingRegressor()


from sklearn.metrics import r2_score,mean_squared_error


def evaluate_model_earn(model,X_earn_train_scaled,X_earn_test_scaled,y_earn_train,y_earn_test):
    model = model.fit(X_earn_train_scaled,y_earn_train)
    pred = model.predict(X_earn_test_scaled)
    r2 = r2_score(y_earn_test,pred)
    mse = mean_squared_error(y_earn_test,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}%; --MSE-- {mse:.2f}%')
    return pred


lr_pred = evaluate_model_earn(lr, X_earn_train_scaled, X_earn_test_scaled, y_earn_train, y_earn_test)
ridge_pred = evaluate_model_earn(ridge, X_earn_train_scaled, X_earn_test_scaled, y_earn_train, y_earn_test)
lasso_pred = evaluate_model_earn(lasso, X_earn_train_scaled, X_earn_test_scaled, y_earn_train, y_earn_test)
rfr_pred = evaluate_model_earn(rfr, X_earn_train_scaled, X_earn_test_scaled, y_earn_train, y_earn_test)
GBR_pred = evaluate_model_earn(GBR, X_earn_train_scaled, X_earn_test_scaled, y_earn_train, y_earn_test)
BR_pred = evaluate_model_earn(BR, X_earn_train_scaled, X_earn_test_scaled, y_earn_train, y_earn_test)










#education

X_educ = df.drop('EDUC',axis=1)
y_educ = df['EDUC']

model_educ = sm.OLS(exog=sm.add_constant(X_educ),endog=y_educ).fit()
print(model_educ.summary())

print(model_educ.params)

X_hours_worked = df.drop('UHRSWORK1',axis=1)
y_hours_worked = df['UHRSWORK1']

education_model = sm.OLS(exog=sm.add_constant(sm.add_constant(X_hours_worked)),endog=y_hours_worked).fit()
print(education_model.summary())
print(education_model.params)
/Users/david/Downloads/cps_00001.csv.gz
