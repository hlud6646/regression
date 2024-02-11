import marimo

__generated_with = "0.2.1"
app = marimo.App()


@app.cell
def __():
    import altair as alt
    import marimo
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_squared_error
    import statsmodels.api as sm
    return (
        KFold,
        PolynomialFeatures,
        alt,
        marimo,
        mean_squared_error,
        np,
        pd,
        sm,
        train_test_split,
    )


@app.cell
def __(pd):
    # Read horrible R file into pandas.
    with open('../data/Auto.data') as f:
        lines = f.readlines()

    header, data = lines[0], lines[1:]
    header = header.split()
    df = []
    for row in data:
        row  = row.split()
        name = "" if len(row) < 9 else " ".join(row[8:])
        row  = row[:8] + [name.strip()[1:-1]]
        df.append(row)

    df = pd.DataFrame(df)
    df.columns = header
    name = df.pop('name')
    df = df.apply(pd.to_numeric, errors='coerce')
    df['name'] = name
    df = df.dropna().reset_index(drop=True)
    df.insert(0, 'intercept', [1,]*len(df))
    df.head(3)
    return data, df, f, header, lines, name, row


@app.cell
def __(pd):
    def poly(v, d):
        "Return a dataframe of polynomial features given a series and a degree."
        return pd.DataFrame({f'{v.name}^{i}': v**i for i in range(1, d+1)})
    return poly,


@app.cell
def __(
    alt,
    df,
    marimo,
    mean_squared_error,
    pd,
    poly,
    sm,
    train_test_split,
):
    # Use the validation method 10 times, each time using a different equal (random) split of the observations into training and validation sets. Plot the resulting estimated test MSEs.

    px = poly(df.horsepower, 8).reset_index(drop=True)
    y  = df.mpg.reset_index(drop=True)

    def trial():
        """
        Using the same train test split, compute the mse for polynomial 
        models with degree 1 to 7."""
        # Full polynomial expansions.
        px_train, px_test, y_train, y_test = train_test_split(px, y, train_size=.5)

        res = []

        for degree in range(1, 8):
            # Select only the columns of the polynomial expansion up to degree
            x_train = px_train.iloc[:, :degree]
            model   = sm.OLS(y_train, x_train)
            results = model.fit()
            y_pred  = results.predict(px_test.iloc[:, :degree])
            res.append(mean_squared_error(y_test, y_pred))
        return res

    results = pd.DataFrame([trial() for _ in range(10)])
    results.columns = [str(d) for d in range(1, 8)]

    # Plot
    df2 = (results.reset_index(names="trial")
            .melt(id_vars='trial', var_name = "degree", value_name = "mse")
          )
    chart = alt.Chart(df2).mark_line(clip=True).encode(
        x='degree',
        y=alt.Y('mse').scale(domain=(0, 100)),
        color='trial:N'
    )
    marimo.ui.altair_chart(chart)
    return chart, df2, px, results, trial, y


@app.cell
def __(alt, marimo, pd, px, sm, y):
    # Compute and plot the LOOCV error curve (as function of the polynomial degree).

    def llocv(degree):
        e = 0
        x = px.iloc[:, :degree+1]

        for j in range(len(x)):
            x_train = x.iloc[[i for i in range(256) if i != j]]
            y_train = y.iloc[[i for i in range(256) if i != j]]
            model  = sm.OLS(y_train, x_train)
            results = model.fit()
            y_pred = results.predict(x.iloc[j])[0]
            e += (y[j] - y_pred)**2
        return e / len(x)

    results2 = pd.DataFrame({
        "degree" : [str(d) for d in range(1, 8)],
        "mse"    : [llocv(d) for d in range(1, 8)]
    })

    chart2 = alt.Chart(results2).mark_line().encode(
        x='degree',
        y='mse',
    )
    chart2 = marimo.ui.altair_chart(chart2)
    chart2
    # Looks different to R output in notes...
    return chart2, llocv, results2


@app.cell
def __(KFold, mean_squared_error, px, sm, y):
    foo = []

    for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=10, shuffle=True).split(px, y)):
        y_train = y[train_idx]
        y_test  = y[test_idx]
        
        for degree in range(1, 8):
            x_train = px.iloc[train_idx, :degree + 1]
            x_test  = px.iloc[test_idx, :degree + 1]
            model   = sm.OLS(y_train, x_train)
            results3 = model.fit()
            y_pred = results3.predict(x_test)
            foo.append((degree, fold, mean_squared_error(y_test, y_pred)))
    return (
        degree,
        fold,
        foo,
        model,
        results3,
        test_idx,
        train_idx,
        x_test,
        x_train,
        y_pred,
        y_test,
        y_train,
    )


@app.cell
def __(alt, foo, marimo, pd):
    chart3 = alt.Chart(pd.DataFrame(foo,  columns = ["degree", "fold", "mse"])).mark_line().encode(
        x='degree:O',
        y=alt.Y('mse').scale(domain=(0, 100)),
        color='fold:N'
    )
    marimo.ui.altair_chart(chart3)
    return chart3,


if __name__ == "__main__":
    app.run()
