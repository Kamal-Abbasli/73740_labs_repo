\# Data Card: UCI Adult Census Dataset



\## 1. Dataset Overview

\- \*\*Name:\*\* UCI Adult Income Dataset (also known as "Census Income").

\- \*\*Source:\*\* Extracted from the 1994 US Census Bureau database by Ronny Kohavi and Barry Becker. Loaded via OpenML (`name='adult'`, version 2).

\- \*\*License:\*\* Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain.

\- \*\*Dataset Size:\*\* 48,842 total rows and 15 columns (14 features + 1 target column).

\- \*\*Target Variable:\*\* `income` (binary classification: `<=50K` vs. `>50K`).

\- \*\*Processed Dataset Fingerprint (SHA-256):\*\* `16c15d72d4ab8d7453a884edb90dccbdb8344d9da918ed9d18359b6c8490a04d`



\## 2. Features Description

\- \*\*Numerical Features (6):\*\* `age`, `fnlwgt` (census weight), `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.

\- \*\*Categorical Features (8):\*\* `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.

\- \*\*Engineered Features (2):\*\*

&#x20; - `capital\_net`: `capital-gain - capital-loss` (net annual investment return).

&#x20; - `hours\_per\_age`: `hours-per-week / (age - 16)` (work intensity relative to working age).



\## 3. Data Quality and Missing Values

\- Missing answers were recorded with the symbol `?`.

\- Missing values only exist in three categorical columns:

&#x20; - `workclass`: 2,799 missing entries (5.73%)

&#x20; - `occupation`: 2,809 missing entries (5.75%)

&#x20; - `native-country`: 857 missing entries (1.75%)

\- All numeric features had 0 missing values and stayed within expected historical boundaries.



\## 4. Known Biases

\- \*\*Gender Imbalance:\*\* Approximately 67% of records represent male individuals due to 1994 household survey practices.

\- \*\*Racial Imbalance:\*\* Around 85% of records represent White individuals, meaning minority groups have significantly fewer samples.

\- \*\*Time/Historical Bias:\*\* Collected in 1994; wage thresholds and economic standards have changed drastically over 30 years.



\## 5. Privacy \& Ethical Concerns

\- The dataset was anonymized by the US Census Bureau through demographic aggregation and removing personal names or street addresses.

\- However, it contains sensitive protected attributes like `race`, `sex`, and `native-country`, meaning any deployed model must be audited for demographic fairness to avoid unfair automated decisions.

