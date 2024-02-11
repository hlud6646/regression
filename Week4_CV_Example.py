import marimo

__generated_with = "0.2.1"
app = marimo.App()


@app.cell
def __():
    import matplotlib.pyplot as plt
    import altair as alt
    import marimo
    import numpy as np
    import pandas as pd
    import polars
    import random
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from tqdm.notebook import tqdm
    return alt, marimo, np, pd, plt, polars, random, sm, smf, tqdm


@app.cell
def __(pd):
    df = pd.read_csv('./data/mos_df.txt', delimiter=' ')
    df.head(3)
    return df,


@app.cell
def __(df, sm):
    # Train the full model, just to make sure we're on the same page as the notes.
    model = sm.GLM(
        endog = df.Maxtemp,
        exog  = sm.add_constant(df.loc[:, df.columns != 'Maxtemp'])
    )
    res = model.fit()
    print(res.summary())
    return model, res


@app.cell
def __(alt, df, marimo, pd, res, sm):
    def check():
        x1     = range(15)
        y_true = df.Maxtemp[:15]
        test_x = sm.add_constant(df.loc[:, df.columns != 'Maxtemp'])[:15]
        y_hat  = res.predict(test_x)

        out = pd.DataFrame({"index": range(15), "true": y_true, "pred": y_hat})

        out = out.melt(id_vars = ["index"], var_name = "kind")
        
        chart = alt.Chart(out).mark_point().encode(
            x='index',
            y='value',
            color='kind',
        )

        chart = marimo.ui.altair_chart(chart)

        
        return(chart)

    check()
    return check,


@app.cell
def __(df, np, random, sm):
    def cv_folds(df, k=10):
        "Generate k (train, test) splits of the dataframe."
        n = len(df)
        chunk_size = n // k
        idx = random.sample(range(n), n)

        for i in range(0, n, chunk_size):
            if i + chunk_size > n:
                break
            train_idx = idx[0: i] + idx[i + chunk_size :]
            test_idx  = idx[i:i + chunk_size] 
            yield df.iloc[train_idx, :], df.iloc[test_idx, :]

    cv_results = []

    for train, test in cv_folds(df):

        x  = sm.add_constant(train.loc[:, train.columns != 'Maxtemp'])
        y = train.Maxtemp

        m = sm.GLM(y, x)
        r = m.fit()

        test_x = sm.add_constant(test.loc[:, test.columns != 'Maxtemp'])
        test_y = test.Maxtemp

        y_hat     = r.predict(test_x)
        residuals = test_y - y_hat 

        cv_results.append(sum(residuals**2) / len(residuals))

    np.mean(cv_results)
    return (
        cv_folds,
        cv_results,
        m,
        r,
        residuals,
        test,
        test_x,
        test_y,
        train,
        x,
        y,
        y_hat,
    )


if __name__ == "__main__":
    app.run()
