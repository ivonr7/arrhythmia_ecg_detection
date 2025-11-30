import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, np, pd, plt, sns


@app.cell
def _(sns):
    sns.set_theme(
        context='poster',
        style='whitegrid'
    )
    return


@app.cell
def _(pd):
    cnn = pd.read_csv("dataset/cnn_acc.csv").pipe(normalize_epochs)
    lstm = pd.read_csv("dataset/lstm_acc.csv").pipe(normalize_epochs)
    fnn = pd.read_csv('dataset/fnn_acc.csv').pipe(normalize_epochs)
    transformer = pd.read_csv('dataset/transformer_acc.csv').pipe(normalize_epochs)
    transformer = (transformer - transformer.min()) / (transformer.max() - transformer.min())
    model_opts = {
        'cnn':cnn,
        'lstm':lstm,
        'fnn': fnn,
        'transformer':transformer
    }
    transformer
    return cnn, fnn, lstm, model_opts, transformer


@app.function
def normalize_epochs(df):
    df['step'] = (df['epoch'] - df['epoch'].min()) / (df['epoch'].max() - df['epoch'].min())
    return df.set_index('step')


@app.cell
def _():
    return


@app.cell
def _(mo, model_opts):
    model = mo.ui.dropdown(options=model_opts.keys(),value='cnn')
    model
    return (model,)


@app.cell
def _(model, model_opts, plt, sns):
    def to_long(df):
        value_cols = [c for c in df.columns if c != 'epoch']

        df_long = df.melt(id_vars='epoch', value_vars=value_cols,
                       var_name='metric', value_name='value')
        return df_long

    if model.value == 'transformer':
        data = (model_opts[model.value]- model_opts[model.value].min()) / (model_opts[model.value].max() - model_opts[model.value].min())
    else:
        data = model_opts[model.value]
    plotting_data = data.pipe(to_long)
    sns.lineplot(data=plotting_data, x='epoch', y='value', hue='metric',alpha = 0.8)
    sns.despine()
    plt.legend(bbox_to_anchor = (1.6,1.1),frameon = False)
    plt.title(f"Training Metrics {model.value.upper()}")
    return


@app.cell
def _(cnn, mo):
    yaxis = mo.ui.dropdown(options=cnn.columns,value='val_acc')
    yaxis
    return (yaxis,)


@app.cell
def _(cnn, fnn, lstm, plt, transformer, yaxis):
    ax = cnn.plot(y = yaxis.value, label = 'CNN')
    fnn.plot(y = yaxis.value, label = 'FNN',ax = ax)
    lstm.plot(y = yaxis.value, label = 'LSTM',ax = ax)
    transformer.plot(y = yaxis.value, label = 'Transformer', ax = ax)
    plt.legend(bbox_to_anchor = (1.5,1.2),ncols = 4,frameon = False)
    plt.ylabel(yaxis.value)
    return


@app.cell
def _():
    import json
    def rd_json(path):
        with open(path) as f:
            return json.load(f)
    return (rd_json,)


@app.cell
def _(pd, rd_json):
    lstm_con = pd.Series(rd_json('dataset/lstm_class_types.json'))
    transformer_con = pd.Series(rd_json('dataset/transformer_class_types.json'))
    transformer_con

    cms = {
        'lstm':lstm_con,
        'transformer': transformer_con
    }
    return (cms,)


@app.cell
def _(np):
    def to_matrix(s):
        return np.array(
            [
                [s['TN'], s['FN']],
                [s['FP'], s['TP']]
            ]
        )
    return (to_matrix,)


@app.cell
def _(cms, mo):
    m_cm = mo.ui.dropdown(options=cms.keys(), value = 'lstm')
    m_cm
    return (m_cm,)


@app.cell
def _(cms, m_cm, plt, sns, to_matrix):
    sns.heatmap(to_matrix(cms[m_cm.value]), annot=True)
    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.title(f"{m_cm.value.upper()} Confusion Matrix")
    return


if __name__ == "__main__":
    app.run()
