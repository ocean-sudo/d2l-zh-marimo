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
    # 自定义层

    深度学习成功背后的一个因素是神经网络的灵活性：
    我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。
    例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。
    有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。
    在这些情况下，必须构建自定义层。本节将展示如何构建自定义层。

    ## 不带参数的层

    首先，我们(**构造一个没有任何参数的自定义层**)。
    回忆一下在 :numref:`sec_model_construction`对块的介绍，
    这应该看起来很眼熟。
    下面的`CenteredLayer`类要从其输入中减去均值。
    要构建它，我们只需继承基础层类并实现前向传播功能。
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from torch import nn


    class CenteredLayer(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, X):
            return X - X.mean()

    return CenteredLayer, F, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们向该层提供一些数据，验证它是否能按预期工作。
    """)
    return


@app.cell
def _(CenteredLayer, torch):
    layer = CenteredLayer()
    layer(torch.FloatTensor([1, 2, 3, 4, 5]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    现在，我们可以[**将层作为组件合并到更复杂的模型中**]。
    """)
    return


@app.cell
def _(CenteredLayer, nn):
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    return (net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0。
    由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。
    """)
    return


@app.cell
def _(net, torch):
    Y = net(torch.rand(4, 8))
    Y.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [**带参数的层**]

    以上我们知道了如何定义简单的层，下面我们继续定义具有参数的层，
    这些参数可以通过训练进行调整。
    我们可以使用内置函数来创建参数，这些函数提供一些基本的管理功能。
    比如管理访问、初始化、共享、保存和加载模型参数。
    这样做的好处之一是：我们不需要为每个自定义层编写自定义的序列化程序。

    现在，让我们实现自定义版本的全连接层。
    回想一下，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。
    在此实现中，我们使用修正线性单元作为激活函数。
    该层需要输入参数：`in_units`和`units`，分别表示输入数和输出数。
    """)
    return


@app.cell
def _(F, nn, torch):
    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_units, units))
            self.bias = nn.Parameter(torch.randn(units,))
        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias.data
            return F.relu(linear)

    return (MyLinear,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    接下来，我们实例化`MyLinear`类并访问其模型参数。
    """)
    return


@app.cell
def _(MyLinear):
    linear = MyLinear(5, 3)
    linear.weight
    return (linear,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们可以[**使用自定义层直接执行前向传播计算**]。
    """)
    return


@app.cell
def _(linear, torch):
    linear(torch.rand(2, 5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们还可以(**使用自定义层构建模型**)，就像使用内置的全连接层一样使用自定义层。
    """)
    return


@app.cell
def _(MyLinear, nn, torch):
    net_1 = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    net_1(torch.rand(2, 64))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    * 我们可以通过基本层类设计自定义层。这允许我们定义灵活的新层，其行为与深度学习框架中的任何现有层不同。
    * 在自定义层定义完成后，我们就可以在任意环境和网络架构中调用该自定义层。
    * 层可以有局部参数，这些参数可以通过内置函数创建。

    ## 练习

    1. 设计一个接受输入并计算张量降维的层，它返回$y_k = \sum_{i, j} W_{ijk} x_i x_j$。
    1. 设计一个返回输入数据的傅立叶系数前半部分的层。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Discussions](https://discuss.d2l.ai/t/1835)
    """)
    return


if __name__ == "__main__":
    app.run()
