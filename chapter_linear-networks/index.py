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
    # 线性神经网络
    :label:`chap_linear`

    在介绍深度神经网络之前，我们需要了解神经网络训练的基础知识。
    本章我们将介绍神经网络的整个训练过程，
    包括：定义简单的神经网络架构、数据处理、指定损失函数和如何训练模型。
    为了更容易学习，我们将从经典算法————*线性*神经网络开始，介绍神经网络的基础知识。
    经典统计学习技术中的线性回归和softmax回归可以视为线性神经网络，
    这些知识将为本书其他部分中更复杂的技术奠定基础。

    :begin_tab:toc
     - [linear-regression](linear-regression.ipynb)
     - [linear-regression-scratch](linear-regression-scratch.ipynb)
     - [linear-regression-concise](linear-regression-concise.ipynb)
     - [softmax-regression](softmax-regression.ipynb)
     - [image-classification-dataset](image-classification-dataset.ipynb)
     - [softmax-regression-scratch](softmax-regression-scratch.ipynb)
     - [softmax-regression-concise](softmax-regression-concise.ipynb)
    :end_tab:
    """)
    return


if __name__ == "__main__":
    app.run()
