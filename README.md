### Install

This project requires Python and the following Python libraries installed:
[NumPy](http://www.numpy.org/)、[Pandas](http://pandas.pydata.org/)、[matplotlib](https://matplotlib.org/)、[scikit-learn](http://scikit-learn.org/stable/)、[LightGBM](https://github.com/Microsoft/LightGBM)、[Kaggle API](https://github.com/Kaggle/kaggle-api)

### Run
1. **Download dataset** from Kaggle and save in dataset/

```bash
kaggle competitions download -c ga-customer-revenue-prediction
```

2. **One-line command** in CLI

```bash
bash ./run.sh
```

3. **Submit to Kaggle**

```bash
kaggle competitions submit ga-customer-revenue-prediction -f submission.csv -m 'I love you, jo4x962k7JL'
kaggle competitions submissions -c ga-customer-revenue-prediction
```
