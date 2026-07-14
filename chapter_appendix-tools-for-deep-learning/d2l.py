import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `d2l` API 文档
    :label:`sec_d2l`

    `d2l`包以下成员的实现及其定义和解释部分可在[源文件](https://github.com/d2l-ai/d2l-en/tree/master/d2l)中找到。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```eval_rst
    .. currentmodule:: d2l.torch
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 模型

    ```eval_rst
    .. autoclass:: Module
       :members:

    .. autoclass:: LinearRegressionScratch
       :members:

    .. autoclass:: LinearRegression
       :members:

    .. autoclass:: Classification
       :members:
    ```

    ## 数据

    ```eval_rst
    .. autoclass:: DataModule
       :members:

    .. autoclass:: SyntheticRegressionData
       :members:

    .. autoclass:: FashionMNIST
       :members:
    ```

    ## 训练

    ```eval_rst
    .. autoclass:: Trainer
       :members:

    .. autoclass:: SGD
       :members:
    ```

    ## 公用

    ```eval_rst
    .. autofunction:: add_to_class

    .. autofunction:: cpu

    .. autofunction:: gpu

    .. autofunction:: num_gpus

    .. autoclass:: ProgressBoard
       :members:

    .. autoclass:: HyperParameters
       :members:
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
