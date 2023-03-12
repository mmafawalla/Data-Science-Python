"""
Function
Purpose: Printing Influential Data Points and Outliers from Dataset
Example: For meeting model assumptions e.g Logistic Regression
Criteria: Observational values printed
Note: a correlation matrix can also be used
Source: https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290
"""

# Use GLM method for logreg here so that we can retrieve the influence measures
logit_results = GLM(y, X_constant, family=families.Binomial()).fit()

# Get influence measures
influence = logit_results.get_influence()

# Obtain summary df of influence measures
summ_df = influence.summary_frame()

# Filter summary df to Cook's distance values only
diagnosis_df = summ_df[['cooks_d']]

# Set Cook's distance threshold
cook_threshold = 4 / len(X)

# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))

# Find observations which are BOTH outlier (std dev > 3) and highly influential
extreme = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) &
                       (diagnosis_df['std_resid'] > 3)]

# Show top 5 highly influential outlier observations
extreme.sort_values("cooks_d", ascending=False).head()

"""
END FUNCTION
"""

"""
Purpose: Calculate Variance Inflation Factor(VIF) for a dataset for multicollinearity
Example: Assumption for logistic regression
Criteria: No value should exceed 5(for showing no multicollinearity)
Source: https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290
"""
# Use variance inflation factor to identify any significant multi-collinearity
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)

calc_vif(X_constant)
"""
END FUNCTION
"""

"""
Purpose: Determine independence of observations
Example: Assumption for logistic regression
Criteria: visual check that residuals should be randomly scattered around the centreline of zero
Source: https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290
"""
# Setup logistic regression model using GLM method so that we can retrieve residuals
logit_results = GLM(y, X_constant, family=families.Binomial()).fit()

# Setup plot
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
                     xlabel="Index Number",
                     ylabel="Deviance Residuals")

# Generate residual series plot using standardized deviance residuals
ax.plot(X.index.tolist(), stats.zscore(logit_results.resid_deviance))

# Draw horizontal line at y=0
plt.axhline(y = 0, ls="--", color='red');
"""
END FUNCTION
"""



"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""



"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""



"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""


"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""


"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""



"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""


"""
Purpose: 
Example: 
Criteria: 
Source:
"""

"""
END FUNCTION
"""

