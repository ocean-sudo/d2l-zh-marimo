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
    # 单发多框检测（SSD）
    :label:`sec_ssd`

    在 :numref:`sec_bbox`— :numref:`sec_object-detection-dataset`中，我们分别介绍了边界框、锚框、多尺度目标检测和用于目标检测的数据集。
    现在我们已经准备好使用这样的背景知识来设计一个目标检测模型：单发多框检测（SSD） :cite:`Liu.Anguelov.Erhan.ea.2016`。
    该模型简单、快速且被广泛使用。尽管这只是其中一种目标检测模型，但本节中的一些设计原则和实现细节也适用于其他模型。

    ## 模型

     :numref:`fig_ssd`描述了单发多框检测模型的设计。
    此模型主要由基础网络组成，其后是几个多尺度特征块。
    基本网络用于从输入图像中提取特征，因此它可以使用深度卷积神经网络。
    单发多框检测论文中选用了在分类层之前截断的VGG :cite:`Liu.Anguelov.Erhan.ea.2016`，现在也常用ResNet替代。
    我们可以设计基础网络，使它输出的高和宽较大。
    这样一来，基于该特征图生成的锚框数量较多，可以用来检测尺寸较小的目标。
    接下来的每个多尺度特征块将上一层提供的特征图的高和宽缩小（如减半），并使特征图中每个单元在输入图像上的感受野变得更广阔。

    回想一下在 :numref:`sec_multiscale-object-detection`中，通过深度神经网络分层表示图像的多尺度目标检测的设计。
    由于接近 :numref:`fig_ssd`顶部的多尺度特征图较小，但具有较大的感受野，它们适合检测较少但较大的物体。
    简而言之，通过多尺度特征块，单发多框检测生成不同大小的锚框，并通过预测边界框的类别和偏移量来检测大小不同的目标，因此这是一个多尺度目标检测模型。

    ![单发多框检测模型主要由一个基础网络块和若干多尺度特征块串联而成。](http://d2l.ai/_images/ssd.svg)
    :label:`fig_ssd`

    在下面，我们将介绍 :numref:`fig_ssd`中不同块的实施细节。
    首先，我们将讨论如何实施类别和边界框预测。

    ### [**类别预测层**]

    设目标类别的数量为$q$。这样一来，锚框有$q+1$个类别，其中0类是背景。
    在某个尺度下，设特征图的高和宽分别为$h$和$w$。
    如果以其中每个单元为中心生成$a$个锚框，那么我们需要对$hwa$个锚框进行分类。
    如果使用全连接层作为输出，很容易导致模型参数过多。
    回忆 :numref:`sec_nin`一节介绍的使用卷积层的通道来输出类别预测的方法，
    单发多框检测采用同样的方法来降低模型复杂度。

    具体来说，类别预测层使用一个保持输入高和宽的卷积层。
    这样一来，输出和输入在特征图宽和高上的空间坐标一一对应。
    考虑输出和输入同一空间坐标（$x$、$y$）：输出特征图上（$x$、$y$）坐标的通道里包含了以输入特征图（$x$、$y$）坐标为中心生成的所有锚框的类别预测。
    因此输出通道数为$a(q+1)$，其中索引为$i(q+1) + j$（$0 \leq j \leq q$）的通道代表了索引为$i$的锚框有关类别索引为$j$的预测。

    在下面，我们定义了这样一个类别预测层，通过参数`num_anchors`和`num_classes`分别指定了$a$和$q$。
    该图层使用填充为1的$3\times3$的卷积层。此卷积层的输入和输出的宽度和高度保持不变。
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as F
    from d2l import torch as d2l


    def cls_predictor(num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                         kernel_size=3, padding=1)

    return F, cls_predictor, d2l, nn, torch, torchvision


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (**边界框预测层**)

    边界框预测层的设计与类别预测层的设计类似。
    唯一不同的是，这里需要为每个锚框预测4个偏移量，而不是$q+1$个类别。
    """)
    return


@app.cell
def _(nn):
    def bbox_predictor(num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    return (bbox_predictor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [**连结多尺度的预测**]

    正如我们所提到的，单发多框检测使用多尺度特征图来生成锚框并预测其类别和偏移量。
    在不同的尺度下，特征图的形状或以同一单元为中心的锚框的数量可能会有所不同。
    因此，不同尺度下预测输出的形状可能会有所不同。

    在以下示例中，我们为同一个小批量构建两个不同比例（`Y1`和`Y2`）的特征图，其中`Y2`的高度和宽度是`Y1`的一半。
    以类别预测为例，假设`Y1`和`Y2`的每个单元分别生成了$5$个和$3$个锚框。
    进一步假设目标类别的数量为$10$，对于特征图`Y1`和`Y2`，类别预测输出中的通道数分别为$5\times(10+1)=55$和$3\times(10+1)=33$，其中任一输出的形状是（批量大小，通道数，高度，宽度）。
    """)
    return


@app.cell
def _(cls_predictor, torch):
    def forward(x, block):
        return block(_x)
    Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
    (Y1.shape, Y2.shape)
    return Y1, Y2, forward


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    正如我们所看到的，除了批量大小这一维度外，其他三个维度都具有不同的尺寸。
    为了将这两个预测输出链接起来以提高计算效率，我们将把这些张量转换为更一致的格式。

    通道维包含中心相同的锚框的预测结果。我们首先将通道维移到最后一维。
    因为不同尺度下批量大小仍保持不变，我们可以将预测结果转成二维的（批量大小，高$\times$宽$\times$通道数）的格式，以方便之后在维度$1$上的连结。
    """)
    return


@app.cell
def _(torch):
    def flatten_pred(pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    def concat_preds(preds):
        return torch.cat([flatten_pred(p) for p in preds], dim=1)

    return (concat_preds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    这样一来，尽管`Y1`和`Y2`在通道数、高度和宽度方面具有不同的大小，我们仍然可以在同一个小批量的两个不同尺度上连接这两个预测输出。
    """)
    return


@app.cell
def _(Y1, Y2, concat_preds):
    concat_preds([Y1, Y2]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [**高和宽减半块**]

    为了在多个尺度下检测目标，我们在下面定义了高和宽减半块`down_sample_blk`，该模块将输入特征图的高度和宽度减半。
    事实上，该块应用了在 :numref:`subsec_vgg-blocks`中的VGG模块设计。
    更具体地说，每个高和宽减半块由两个填充为$1$的$3\times3$的卷积层、以及步幅为$2$的$2\times2$最大汇聚层组成。
    我们知道，填充为$1$的$3\times3$卷积层不改变特征图的形状。但是，其后的$2\times2$的最大汇聚层将输入特征图的高度和宽度减少了一半。
    对于此高和宽减半块的输入和输出特征图，因为$1\times 2+(3-1)+(3-1)=6$，所以输出中的每个单元在输入上都有一个$6\times6$的感受野。因此，高和宽减半块会扩大每个单元在其输出特征图中的感受野。
    """)
    return


@app.cell
def _(nn):
    def down_sample_blk(in_channels, out_channels):
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels, out_channels,
                                 kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(2))
        return nn.Sequential(*blk)

    return (down_sample_blk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    在以下示例中，我们构建的高和宽减半块会更改输入通道的数量，并将输入特征图的高度和宽度减半。
    """)
    return


@app.cell
def _(down_sample_blk, forward, torch):
    forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [**基本网络块**]

    基本网络块用于从输入图像中抽取特征。
    为了计算简洁，我们构造了一个小的基础网络，该网络串联3个高和宽减半块，并逐步将通道数翻倍。
    给定输入图像的形状为$256\times256$，此基本网络块输出的特征图形状为$32 \times 32$（$256/2^3=32$）。
    """)
    return


@app.cell
def _(down_sample_blk, forward, nn, torch):
    def base_net():
        blk = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters) - 1):
            blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
        return nn.Sequential(*blk)

    forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
    return (base_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 完整的模型

    [**完整的单发多框检测模型由五个模块组成**]。每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量。在这五个模块中，第一个是基本网络块，第二个到第四个是高和宽减半块，最后一个模块使用全局最大池将高度和宽度都降到1。从技术上讲，第二到第五个区块都是 :numref:`fig_ssd`中的多尺度特征块。
    """)
    return


@app.cell
def _(base_net, down_sample_blk, nn):
    def get_blk(i):
        if i == 0:
            blk = base_net()
        elif i == 1:
            blk = down_sample_blk(64, 128)
        elif i == 4:
            blk = nn.AdaptiveMaxPool2d((1,1))
        else:
            blk = down_sample_blk(128, 128)
        return blk

    return (get_blk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    现在我们[**为每个块定义前向传播**]。与图像分类任务不同，此处的输出包括：CNN特征图`Y`；在当前尺度下根据`Y`生成的锚框；预测的这些锚框的类别和偏移量（基于`Y`）。
    """)
    return


@app.cell
def _(d2l):
    def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        _anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
        _cls_preds = cls_predictor(Y)
        _bbox_preds = bbox_predictor(Y)
        return (Y, _anchors, _cls_preds, _bbox_preds)

    return (blk_forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    回想一下，在 :numref:`fig_ssd`中，一个较接近顶部的多尺度特征块是用于检测较大目标的，因此需要生成更大的锚框。
    在上面的前向传播中，在每个多尺度特征块上，我们通过调用的`multibox_prior`函数（见 :numref:`sec_anchor`）的`sizes`参数传递两个比例值的列表。
    在下面，0.2和1.05之间的区间被均匀分成五个部分，以确定五个模块的在不同尺度下的较小值：0.2、0.37、0.54、0.71和0.88。
    之后，他们较大的值由$\sqrt{0.2 \times 0.37} = 0.272$、$\sqrt{0.37 \times 0.54} = 0.447$等给出。

    [~~超参数~~]
    """)
    return


@app.cell
def _():
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    return num_anchors, ratios, sizes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    现在，我们就可以按如下方式[**定义完整的模型**]`TinySSD`了。
    """)
    return


@app.cell
def _(
    bbox_predictor,
    blk_forward,
    cls_predictor,
    concat_preds,
    get_blk,
    nn,
    num_anchors,
    ratios,
    sizes,
    torch,
):
    class TinySSD(nn.Module):

        def __init__(self, num_classes, **kwargs):
            super(TinySSD, self).__init__(**kwargs)
            self.num_classes = num_classes
            idx_to_in_channels = [64, 128, 128, 128, 128]
            for i in range(5):  # 即赋值语句self.blk_i=get_blk(i)
                setattr(self, f'blk_{i}', get_blk(i))
                setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
                setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

        def forward(self, X):
            _anchors, _cls_preds, _bbox_preds = ([None] * 5, [None] * 5, [None] * 5)
            for i in range(5):
                X, _anchors[i], _cls_preds[i], _bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i], getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
            _anchors = torch.cat(_anchors, dim=1)
            _cls_preds = concat_preds(_cls_preds)  # getattr(self,'blk_%d'%i)即访问self.blk_i
            _cls_preds = _cls_preds.reshape(_cls_preds.shape[0], -1, self.num_classes + 1)
            _bbox_preds = concat_preds(_bbox_preds)
            return (_anchors, _cls_preds, _bbox_preds)

    return (TinySSD,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们[**创建一个模型实例，然后使用它**]对一个$256 \times 256$像素的小批量图像`X`(**执行前向传播**)。

    如本节前面部分所示，第一个模块输出特征图的形状为$32 \times 32$。
    回想一下，第二到第四个模块为高和宽减半块，第五个模块为全局汇聚层。
    由于以特征图的每个单元为中心有$4$个锚框生成，因此在所有五个尺度下，每个图像总共生成$(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$个锚框。
    """)
    return


@app.cell
def _(TinySSD, torch):
    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    _anchors, _cls_preds, _bbox_preds = net(X)
    print('output anchors:', _anchors.shape)
    print('output class preds:', _cls_preds.shape)
    print('output bbox preds:', _bbox_preds.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 训练模型

    现在，我们将描述如何训练用于目标检测的单发多框检测模型。

    ### 读取数据集和初始化

    首先，让我们[**读取**] :numref:`sec_object-detection-dataset`中描述的(**香蕉检测数据集**)。
    """)
    return


@app.cell
def _(d2l):
    batch_size = 32
    train_iter, _ = d2l.load_data_bananas(batch_size)
    return (train_iter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    香蕉检测数据集中，目标的类别数为1。
    定义好模型后，我们需要(**初始化其参数并定义优化算法**)。
    """)
    return


@app.cell
def _(TinySSD, d2l, torch):
    device, net_1 = (d2l.try_gpu(), TinySSD(num_classes=1))
    trainer = torch.optim.SGD(net_1.parameters(), lr=0.2, weight_decay=0.0005)
    return device, net_1, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [**定义损失函数和评价函数**]

    目标检测有两种类型的损失。
    第一种有关锚框类别的损失：我们可以简单地复用之前图像分类问题里一直使用的交叉熵损失函数来计算；
    第二种有关正类锚框偏移量的损失：预测偏移量是一个回归问题。
    但是，对于这个回归问题，我们在这里不使用 :numref:`subsec_normal_distribution_and_squared_loss`中描述的平方损失，而是使用$L_1$范数损失，即预测值和真实值之差的绝对值。
    掩码变量`bbox_masks`令负类锚框和填充锚框不参与损失的计算。
    最后，我们将锚框类别和偏移量的损失相加，以获得模型的最终损失函数。
    """)
    return


@app.cell
def _(nn):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size, num_classes = (_cls_preds.shape[0], _cls_preds.shape[2])
        cls = cls_loss(_cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(_bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox

    return (calc_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们可以沿用准确率评价分类结果。
    由于偏移量使用了$L_1$范数损失，我们使用*平均绝对误差*来评价边界框的预测结果。这些预测结果是从生成的锚框及其预测偏移量中获得的。
    """)
    return


@app.cell
def _(torch):
    def cls_eval(cls_preds, cls_labels):
        return float((_cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())  # 由于类别预测结果放在最后一维，argmax需要指定最后一维。

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        return float(torch.abs((bbox_labels - _bbox_preds) * bbox_masks).sum())

    return bbox_eval, cls_eval


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [**训练模型**]

    在训练模型时，我们需要在模型的前向传播过程中生成多尺度锚框（`anchors`），并预测其类别（`cls_preds`）和偏移量（`bbox_preds`）。
    然后，我们根据标签信息`Y`为生成的锚框标记类别（`cls_labels`）和偏移量（`bbox_labels`）。
    最后，我们根据类别和偏移量的预测和标注值计算损失函数。为了代码简洁，这里没有评价测试数据集。
    """)
    return


@app.cell
def _(bbox_eval, calc_loss, cls_eval, d2l, device, net_1, train_iter, trainer):
    num_epochs, timer = (20, d2l.Timer())
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    net_2 = net_1.to(device)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net_2.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X_1, Y = (features.to(device), target.to(device))
            _anchors, _cls_preds, _bbox_preds = net_2(X_1)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(_anchors, Y)
            _l = calc_loss(_cls_preds, cls_labels, _bbox_preds, bbox_labels, bbox_masks)
            _l.mean().backward()
            trainer.step()
            metric.add(cls_eval(_cls_preds, cls_labels), cls_labels.numel(), bbox_eval(_bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
        cls_err, bbox_mae = (1 - metric[0] / metric[1], metric[2] / metric[3])
        animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')
    return (net_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [**预测目标**]

    在预测阶段，我们希望能把图像里面所有我们感兴趣的目标检测出来。在下面，我们读取并调整测试图像的大小，然后将其转成卷积层需要的四维格式。
    """)
    return


@app.cell
def _(torchvision):
    X_2 = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
    img = X_2.squeeze(0).permute(1, 2, 0).long()
    return X_2, img


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    使用下面的`multibox_detection`函数，我们可以根据锚框及其预测偏移量得到预测边界框。然后，通过非极大值抑制来移除相似的预测边界框。
    """)
    return


@app.cell
def _(F, X_2, d2l, device, net_2):
    def predict(X):
        net_2.eval()
        _anchors, _cls_preds, _bbox_preds = net_2(X.to(device))
        cls_probs = F.softmax(_cls_preds, dim=2).permute(0, 2, 1)
        output = d2l.multibox_detection(cls_probs, _bbox_preds, _anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]
    output = predict(X_2)
    return (output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    最后，我们[**筛选所有置信度不低于0.9的边界框，做为最终输出**]。
    """)
    return


@app.cell
def _(d2l, img, output, torch):
    def display(img, output, threshold):
        d2l.set_figsize((5, 5))
        fig = d2l.plt.imshow(img)
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            h, w = img.shape[0:2]
            bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
            d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

    display(img, output.cpu(), threshold=0.9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    * 单发多框检测是一种多尺度目标检测模型。基于基础网络块和各个多尺度特征块，单发多框检测生成不同数量和不同大小的锚框，并通过预测这些锚框的类别和偏移量检测不同大小的目标。
    * 在训练单发多框检测模型时，损失函数是根据锚框的类别和偏移量的预测及标注值计算得出的。

    ## 练习

    1. 能通过改进损失函数来改进单发多框检测吗？例如，将预测偏移量用到的$L_1$范数损失替换为平滑$L_1$范数损失。它在零点附近使用平方函数从而更加平滑，这是通过一个超参数$\sigma$来控制平滑区域的：

    $$
    f(x) =
        \begin{cases}
        (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
        |x|-0.5/\sigma^2,& \text{otherwise}
        \end{cases}
    $$

    当$\sigma$非常大时，这种损失类似于$L_1$范数损失。当它的值较小时，损失函数较平滑。
    """)
    return


@app.cell
def _(d2l, torch):
    def smooth_l1(data, scalar):
        out = []
        for i in data:
            if abs(i) < 1 / scalar ** 2:
                out.append((scalar * i) ** 2 / 2)
            else:
                out.append(abs(i) - 0.5 / scalar ** 2)
        return torch.tensor(out)
    sigmas = [10, 1, 0.5]
    lines = ['-', '--', '-.']
    _x = torch.arange(-2, 2, 0.1)
    d2l.set_figsize()
    for _l, s in zip(lines, sigmas):
        _y = smooth_l1(_x, scalar=s)
        d2l.plt.plot(_x, _y, _l, label='sigma=%.1f' % s)
    d2l.plt.legend()
    return (lines,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    此外，在类别预测时，实验中使用了交叉熵损失：设真实类别$j$的预测概率是$p_j$，交叉熵损失为$-\log p_j$。我们还可以使用焦点损失 :cite:`Lin.Goyal.Girshick.ea.2017`。给定超参数$\gamma > 0$和$\alpha > 0$，此损失的定义为：

    $$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

    可以看到，增大$\gamma$可以有效地减少正类预测概率较大时（例如$p_j > 0.5$）的相对损失，因此训练可以更集中在那些错误分类的困难示例上。
    """)
    return


@app.cell
def _(d2l, lines, torch):
    def focal_loss(gamma, x):
        return -(1 - _x) ** gamma * torch.log(_x)
    _x = torch.arange(0.01, 1, 0.01)
    for _l, gamma in zip(lines, [0, 1, 5]):
        _y = d2l.plt.plot(_x, focal_loss(gamma, _x), _l, label='gamma=%.1f' % gamma)
    d2l.plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. 由于篇幅限制，我们在本节中省略了单发多框检测模型的一些实现细节。能否从以下几个方面进一步改进模型：
        1. 当目标比图像小得多时，模型可以将输入图像调大；
        1. 通常会存在大量的负锚框。为了使类别分布更加平衡，我们可以将负锚框的高和宽减半；
        1. 在损失函数中，给类别损失和偏移损失设置不同比重的超参数；
        1. 使用其他方法评估目标检测模型，例如单发多框检测论文 :cite:`Liu.Anguelov.Erhan.ea.2016`中的方法。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Discussions](https://discuss.d2l.ai/t/3204)
    """)
    return


if __name__ == "__main__":
    app.run()
