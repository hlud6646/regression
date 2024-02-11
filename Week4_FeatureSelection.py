import marimo

__generated_with = "0.2.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from sklearn.feature_selection import SequentialFeatureSelector
    return SequentialFeatureSelector, mo


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
