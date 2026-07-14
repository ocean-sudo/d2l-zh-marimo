# /// script
# dependencies = ["d2l-zh@release @ git+https://github.com/d2l-ai/d2l-zh@release"]
# ///

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
    The following additional libraries are needed to run this
    notebook. Note that running on Colab is experimental, please report a Github
    issue if you have any problem.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: git+https://github.com/d2l-ai/d2l-zh@release !pip install git+https://github.com/d2l-ai/d2l-zh@release
    # installing d2l
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 多层感知机的简洁实现
    :label:`sec_mlp_concise`

    本节将介绍(**通过高级API更简洁地实现多层感知机**)。
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn
    from d2l import torch as d2l

    return d2l, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 模型

    与softmax回归的简洁实现（ :numref:`sec_softmax_concise`）相比，
    唯一的区别是我们添加了2个全连接层（之前我们只添加了1个全连接层）。
    第一层是[**隐藏层**]，它(**包含256个隐藏单元，并使用了ReLU激活函数**)。
    第二层是输出层。
    """)
    return


@app.cell
def _(nn):
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);
    return (net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [**训练过程**]的实现与我们实现softmax回归时完全相同，
    这种模块化设计使我们能够将与模型架构有关的内容独立出来。
    """)
    return


@app.cell
def _(net, nn, torch):
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    return batch_size, loss, num_epochs, trainer


@app.cell
def _(batch_size, d2l, loss, net, num_epochs, trainer):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    * 我们可以使用高级API更简洁地实现多层感知机。
    * 对于相同的分类问题，多层感知机的实现与softmax回归的实现相同，只是多层感知机的实现里增加了带有激活函数的隐藏层。

    ## 练习

    1. 尝试添加不同数量的隐藏层（也可以修改学习率），怎么样设置效果最好？
    1. 尝试不同的激活函数，哪个效果最好？
    1. 尝试不同的方案来初始化权重，什么方法效果最好？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Discussions](https://discuss.d2l.ai/t/1802)
    """)
    return


if __name__ == "__main__":
    app.run()
