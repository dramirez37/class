# Applied Labor Economics and Microdata Portfolio

This repository contains applied labor-economics analyses written in Python using CPS/IPUMS-style microdata. The work focuses on building reproducible data-cleaning pipelines, recoding survey variables, constructing wage and labor-market measures, and generating tables/figures for applied econometric interpretation.

The repository is best presented as an **applied econometrics and labor microdata portfolio** rather than as a single fiscal/property-tax project.

## Project overview

### 1. Fertility and labor supply

**Script:** `Code/analysis`

This analysis studies labor-force participation by gender and parental status. It loads CPS-style variables for sex, age, year, labor-force status, number of children under five, and survey weights, then constructs labor-force participation measures and visualizes participation trends by sex and family status.

Key skills demonstrated:

- Microdata loading and variable selection with `pandas`
- Survey-weighted labor-force participation construction
- Grouped time-series summaries by year, sex, and parental status
- Labor-market visualization with `matplotlib` and `seaborn`

### 2. Demographics of the low-wage workforce

**Script:** `Code/analysis2`

This analysis constructs an hourly wage measure from weekly earnings, usual hours worked, and reported hourly wages, then identifies low-wage workers and studies their composition by age, race/ethnicity, education, and gender.

Key skills demonstrated:

- Wage-variable cleaning and validity filtering
- Race/ethnicity and education recoding
- Construction of a low-wage-worker indicator
- Grouped demographic composition analysis
- Visualization of labor-market inequality patterns

### 3. Compensating differentials and night-shift work

**Script:** `Code/analysis3`

This analysis studies irregular shift work and wage differences. It filters working-age employees, constructs log hourly wages, identifies irregular-shift status, recodes education and age groups, and generates tables by education, age, and selected occupations.

Key skills demonstrated:

- Applied labor-economics data cleaning
- Log wage construction
- Shift-work indicator design
- Occupation-level grouping and summary-table generation
- Export of reproducible CSV tables for reporting

### 4. Wage differences across college majors

**Script:** `Code/analysis4`

This analysis studies earnings differences across college majors for young college graduates. It maps degree-field codes to interpretable major categories, filters a 2019 sample, computes weighted log annual earnings, separates Economics from other social sciences, and constructs gender-gap and major-composition summaries.

Key skills demonstrated:

- Degree-field code mapping and categorical recoding
- Weighted earnings summaries by major
- Gender-gap construction by field of study
- Sample restrictions and minimum-cell-size filters
- Export of earnings tables and visualization outputs

### 5. Mincer earnings function over time

**Script:** `Code/analysis5`

This analysis estimates Mincer-style earnings regressions by year and sex over a long historical window. It constructs log weekly earnings, maps education codes into approximate years of schooling, computes potential experience and squared experience, and tracks estimated returns to schooling and experience from 1964 through 2022.

Key skills demonstrated:

- Long-horizon microdata panel construction
- Education-to-schooling recoding
- Potential-experience feature engineering
- Year-by-year regression estimation with `statsmodels`
- Visualization of returns to schooling and experience over time

### 6. Logistic regression and ROC/AUC classification exercise

**Script:** `Code2/problemset4.py`

This exercise trains a logistic regression classifier, creates a train/test split, computes predicted probabilities, and evaluates classification performance with an ROC curve and AUC score.

Key skills demonstrated:

- Binary classification with `scikit-learn`
- Train/test splitting
- ROC curve construction
- AUC-based model evaluation

## Skills demonstrated

- Python data analysis with `pandas`, `NumPy`, `statsmodels`, `scikit-learn`, `matplotlib`, and `seaborn`
- CPS/IPUMS-style microdata cleaning and feature construction
- Labor-market variable recoding and survey-data preparation
- Wage, education, demographic, and labor-force participation analysis
- Regression-based applied econometrics
- Reproducible table and figure generation

## Suggested resume framing

```latex
\resumeSubheading{Applied Econometrics and Labor Microdata Portfolio}{Fall 2024}{Python, pandas, statsmodels, IPUMS/CPS Microdata, Data Visualization}{}
  \begin{itemize}[leftmargin=0.25in]
    \resumeItem{Built a portfolio of applied econometric analyses using large CPS/IPUMS-style microdata extracts, including labor-force participation, low-wage workforce composition, night-shift wage differentials, college-major earnings, and Mincer earnings-function trends.}
    \resumeItem{Cleaned and recoded survey variables across age, gender, education, race, occupation, wages, hours, and survey weights to construct reproducible analysis panels and weighted descriptive statistics.}
    \resumeItem{Estimated yearly Mincer-style earnings regressions from 1964--2022 using Python and statsmodels to track returns to schooling and experience over time by gender.}
    \resumeItem{Generated reporting-ready tables and visualizations for wage distributions, demographic composition, earnings gaps, and labor-market participation patterns.}
  \end{itemize}
```

## Reproducibility notes

The scripts expect CPS/IPUMS-style input files in local `Data/` or `Data2/` folders and save generated tables or figures into `Graphs/` or `Graphs2/`. File paths may need to be adjusted before running outside the original Codespaces/workspace environment.

Recommended workflow:

1. Create a Python environment.
2. Install the dependencies listed below.
3. Place the required microdata extracts in the expected data folders.
4. Run the analysis scripts from the repository root or update paths to match your local environment.

## Dependencies

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
```

## Repository status

This repository is a coursework and applied-analysis portfolio. It emphasizes data cleaning, reproducible analysis panels, labor-economics interpretation, and regression/visualization workflows. It is not intended to be a production software package.
