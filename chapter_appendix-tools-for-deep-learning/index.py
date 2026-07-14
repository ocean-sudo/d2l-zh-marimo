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
    # 附录：深度学习工具
    :label:`chap_appendix_tools`

    为了充分利用《动手学深度学习》，本书将在本附录中介绍不同工具，
    例如如何运行这本交互式开源书籍和为本书做贡献。

    :begin_tab:toc
     - [jupyter](jupyter.ipynb)
     - [sagemaker](sagemaker.ipynb)
     - [aws](aws.ipynb)
     - [selecting-servers-gpus](selecting-servers-gpus.ipynb)
     - [contributing](contributing.ipynb)
     - [d2l](d2l.ipynb)
    :end_tab:
    """)
    return


if __name__ == "__main__":
    app.run()
