\# Lab 04 Report: Cleaning the Census Data and Building a Safe Pipeline



\## 1. Problem Definition

The task in this lab is to take census data and predict if a person makes more than $50,000 a year (`>50K`) or not (`<=50K`) using 14 personal details like age, education, and job.



\- \*\*What we predict:\*\* Income level (`<=50K` vs `>50K`).

\- \*\*How we score it:\*\* Accuracy and F1 score. We need F1 score because about 75% of people in this dataset make under 50K. If a model just guessed "under 50K" every time, it would look 75% accurate while actually being completely useless.

\- \*\*Main rules:\*\* Zero data leakage (no peeking at the test exam early) and making the entire cleaning process run with a single command.



\## 2. Reading the Data and Checking for Errors



\### Downloading the raw data:

We loaded the dataset straight from OpenML without touching or changing anything.

\- Total rows: 48,842

\- Total columns: 15 (14 features and 1 income answer)



\### Checking for impossible numbers:

We checked if any numbers went outside normal human limits:

\- `age` (17 to 90): 0 bad rows.

\- `hours-per-week` (1 to 99): 0 bad rows.

\- `capital-gain` (0 to 99,999): 0 bad rows.

\- `capital-loss` (0 to 4,356): 0 bad rows.



\### Missing answers:

People left answers blank using a question mark (`?`). Only three columns had missing answers:

\- `workclass`: 2,799 people skipped this (5.73%)

\- `occupation`: 2,809 people skipped this (5.75%)

\- `native-country`: 857 people skipped this (1.75%)

We kept all rows and logged these counts instead of deleting them.



\## 3. Checking for Label Conflicts

\- \*\*Conflicting duplicates:\*\* We checked for identical rows with opposite income labels. We found \*\*5 groups of identical profiles\*\* where one copy said `<=50K` and another copy said `>50K`. This shows that real survey data has human errors or missing background information that wasn't recorded.

\- \*\*Manual sample:\*\* We saved 50 random rows into `results/label\_audit\_sample\_50.csv` to inspect the values directly.



\## 4. Cleaning the Data Without Cheating (No Leakage)



To make sure the model never cheats:

1\. \*\*Split First:\*\* We cut the dataset into 70% Training (34,189 people) and 30% Testing (14,653 people) before doing any cleaning.

2\. \*\*Train-Only Rules:\*\* We filled missing numbers using the median from the training set, and missing categories using the most common answer from the training set. The test set was cleaned using those same training numbers so it stayed completely unseen.

3\. \*\*Outliers:\*\* We clipped extreme numbers to the 1st and 99th percentiles calculated only from the training set.

4\. \*\*Duplicates:\*\* We dropped 24 exact duplicate rows from the training set, but left the test set untouched.

5\. \*\*Saved preprocessor:\*\* We saved the fitted cleaning tool to `results/preprocessor.joblib`.



\## 5. Adding New Features and Testing the Baseline



We created two new combination clues:

\- `capital\_net` = `capital-gain - capital-loss` (total profit from investments).

\- `hours\_per\_age` = `hours-per-week / (age - 16)` (how much someone works compared to their adult years).



We tested a Logistic Regression model to see if these new clues helped:

\- \*\*Without new clues:\*\* Accuracy = 85.59%, F1 Score = 0.6658

\- \*\*With new clues:\*\* Accuracy = 85.85%, F1 Score = 0.6712



Both scores went up because net capital profit gives a clearer financial signal than two separate gain/loss columns, and work hours divided by age helps identify high-earning work intensity.



\## 6. Proving It Runs Cleanly

\- The entire process runs from start to finish with: `python lab04\\src\\pipeline.py`.

\- The cleaned dataset has this exact SHA-256 fingerprint:

&#x20; `16c15d72d4ab8d7453a884edb90dccbdb8344d9da918ed9d18359b6c8490a04d`

\- All settings and final test scores are saved in `results/config.yaml`.

