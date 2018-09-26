### Notes
- sim1: Refactor YouHan's code[1], split into data-preprocessing and model-tuning
datagen.py -> sim1.py
- sim2: Use LGBMRegressor to replace sim1's model
datagen.py -> sim2.py
- sim3: combine YouHan's and Olivier's feature engineering, Public LB: 1.5327 -> 1.5176
gen\_olivier.py -> data\_combine.py -> sim3.py

### to-do list
- [ ] Why the output does not shows correct RMSE?


### Reference
1. Deal with json
https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
2. Drop outlier columns
https://www.kaggle.com/ogakulov/feature-engineering-step-by-step
3. YouHan's feature engineering
https://www.kaggle.com/youhanlee/stratified-sampling-for-regression-lb-1-6595
4. Olivier's feature engineering
https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480/code

