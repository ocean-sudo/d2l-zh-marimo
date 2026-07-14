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
    # 异步计算
    :label:`sec_async`

    今天的计算机是高度并行的系统，由多个CPU核、多个GPU、多个处理单元组成。通常每个CPU核有多个线程，每个设备通常有多个GPU，每个GPU有多个处理单元。总之，我们可以同时处理许多不同的事情，并且通常是在不同的设备上。不幸的是，Python并不善于编写并行和异步代码，至少在没有额外帮助的情况下不是好选择。归根结底，Python是单线程的，将来也是不太可能改变的。因此在诸多的深度学习框架中，MXNet和TensorFlow之类则采用了一种*异步编程*（asynchronous programming）模型来提高性能，而PyTorch则使用了Python自己的调度器来实现不同的性能权衡。对PyTorch来说GPU操作在默认情况下是异步的。当调用一个使用GPU的函数时，操作会排队到特定的设备上，但不一定要等到以后才执行。这允许我们并行执行更多的计算，包括在CPU或其他GPU上的操作。

    因此，了解异步编程是如何工作的，通过主动地减少计算需求和相互依赖，有助于我们开发更高效的程序。这能够减少内存开销并提高处理器利用率。
    """)
    return


@app.cell
def _():
    import os
    import subprocess
    import numpy
    import torch
    from torch import nn
    from d2l import torch as d2l

    return d2l, numpy, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 通过后端异步处理
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    作为热身，考虑一个简单问题：生成一个随机矩阵并将其相乘。让我们在NumPy和PyTorch张量中都这样做，看看它们的区别。请注意，PyTorch的`tensor`是在GPU上定义的。
    """)
    return


@app.cell
def _(d2l, numpy, torch):
    # GPU计算热身
    device = d2l.try_gpu()
    _a = torch.randn(size=(1000, 1000), device=device)
    _b = torch.mm(_a, _a)
    with d2l.Benchmark('numpy'):
        for _ in range(10):
            _a = numpy.random.normal(size=(1000, 1000))
            _b = numpy.dot(_a, _a)
    with d2l.Benchmark('torch'):
        for _ in range(10):
            _a = torch.randn(size=(1000, 1000), device=device)
            _b = torch.mm(_a, _a)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    通过PyTorch的基准输出比较快了几个数量级。NumPy点积是在CPU上执行的，而PyTorch矩阵乘法是在GPU上执行的，后者的速度要快得多。但巨大的时间差距表明一定还有其他原因。默认情况下，GPU操作在PyTorch中是异步的。强制PyTorch在返回之前完成所有计算，这种强制说明了之前发生的情况：计算是由后端执行，而前端将控制权返回给了Python。
    """)
    return


@app.cell
def _(d2l, device, torch):
    with d2l.Benchmark():
        for _ in range(10):
            _a = torch.randn(size=(1000, 1000), device=device)
            _b = torch.mm(_a, _a)
        torch.cuda.synchronize(device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    广义上说，PyTorch有一个用于与用户直接交互的前端（例如通过Python），还有一个由系统用来执行计算的后端。如 :numref:`fig_frontends`所示，用户可以用各种前端语言编写PyTorch程序，如Python和C++。不管使用的前端编程语言是什么，PyTorch程序的执行主要发生在C++实现的后端。由前端语言发出的操作被传递到后端执行。后端管理自己的线程，这些线程不断收集和执行排队的任务。请注意，要使其工作，后端必须能够跟踪计算图中各个步骤之间的依赖关系。因此，不可能并行化相互依赖的操作。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![编程语言前端和深度学习框架后端](../img/frontends.png)
    :width:`300px`
    :label:`fig_frontends`

    接下来看看另一个简单例子，以便更好地理解依赖关系图。
    """)
    return


@app.cell
def _(device, torch):
    x = torch.ones((1, 2), device=device)
    y = torch.ones((1, 2), device=device)
    z = x * y + 2
    z
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![后端跟踪计算图中各个步骤之间的依赖关系](http://d2l.ai/_images/asyncgraph.svg)
    :label:`fig_asyncgraph`

    上面的代码片段在 :numref:`fig_asyncgraph`中进行了说明。每当Python前端线程执行前三条语句中的一条语句时，它只是将任务返回到后端队列。当最后一个语句的结果需要被打印出来时，Python前端线程将等待C++后端线程完成变量`z`的结果计算。这种设计的一个好处是Python前端线程不需要执行实际的计算。因此，不管Python的性能如何，对程序的整体性能几乎没有影响。 :numref:`fig_threading`演示了前端和后端如何交互。

    ![前端和后端的交互](http://d2l.ai/_images/threading.svg)
    :label:`fig_threading`

    ## 障碍器与阻塞器
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 改进计算
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Python前端线程和C++后端线程之间的简化交互可以概括如下：

    1. 前端命令后端将计算任务`y = x + 1`插入队列；
    1. 然后后端从队列接收计算任务并执行；
    1. 然后后端将计算结果返回到前端。

    假设这三个阶段的持续时间分别为$t_1, t_2, t_3$。如果不使用异步编程，执行10000次计算所需的总时间约为$10000 (t_1+ t_2 + t_3)$。如果使用异步编程，因为前端不必等待后端为每个循环返回计算结果，执行$10000$次计算所花费的总时间可以减少到$t_1 + 10000 t_2 + t_3$（假设$10000 t_2 > 9999t_1$）。

    ## 小结

    * 深度学习框架可以将Python前端的控制与后端的执行解耦，使得命令可以快速地异步插入后端、并行执行。
    * 异步产生了一个相当灵活的前端，但请注意：过度填充任务队列可能会导致内存消耗过多。建议对每个小批量进行同步，以保持前端和后端大致同步。
    * 芯片供应商提供了复杂的性能分析工具，以获得对深度学习效率更精确的洞察。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 练习
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. 在CPU上，对本节中相同的矩阵乘法操作进行基准测试，仍然可以通过后端观察异步吗？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Discussions](https://discuss.d2l.ai/t/2791)
    """)
    return


if __name__ == "__main__":
    app.run()
