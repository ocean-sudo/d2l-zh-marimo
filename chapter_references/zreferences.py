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
    ```eval_rst
    .. only:: html

       参考文献
       ==========
    ```

    :bibliography:`../d2l.bib`
    """)
    return


if __name__ == "__main__":
    app.run()
