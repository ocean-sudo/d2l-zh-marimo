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
    《动手学深度学习》
    ========================

    ```eval_rst
    .. raw:: html
       :file: frontpage.html
    ```

    :begin_tab:toc
     - [chapter_preface/index](chapter_preface/index.ipynb)
     - [chapter_installation/index](chapter_installation/index.ipynb)
     - [chapter_notation/index](chapter_notation/index.ipynb)
    :end_tab:

    :begin_tab:toc
     - [chapter_introduction/index](chapter_introduction/index.ipynb)
     - [chapter_preliminaries/index](chapter_preliminaries/index.ipynb)
     - [chapter_linear-networks/index](chapter_linear-networks/index.ipynb)
     - [chapter_multilayer-perceptrons/index](chapter_multilayer-perceptrons/index.ipynb)
     - [chapter_deep-learning-computation/index](chapter_deep-learning-computation/index.ipynb)
     - [chapter_convolutional-neural-networks/index](chapter_convolutional-neural-networks/index.ipynb)
     - [chapter_convolutional-modern/index](chapter_convolutional-modern/index.ipynb)
     - [chapter_recurrent-neural-networks/index](chapter_recurrent-neural-networks/index.ipynb)
     - [chapter_recurrent-modern/index](chapter_recurrent-modern/index.ipynb)
     - [chapter_attention-mechanisms/index](chapter_attention-mechanisms/index.ipynb)
     - [chapter_optimization/index](chapter_optimization/index.ipynb)
     - [chapter_computational-performance/index](chapter_computational-performance/index.ipynb)
     - [chapter_computer-vision/index](chapter_computer-vision/index.ipynb)
     - [chapter_natural-language-processing-pretraining/index](chapter_natural-language-processing-pretraining/index.ipynb)
     - [chapter_natural-language-processing-applications/index](chapter_natural-language-processing-applications/index.ipynb)
     - [chapter_appendix-tools-for-deep-learning/index](chapter_appendix-tools-for-deep-learning/index.ipynb)
    :end_tab:

    :begin_tab:toc
     - [chapter_references/zreferences](chapter_references/zreferences.ipynb)
    :end_tab:
    """)
    return


if __name__ == "__main__":
    app.run()
