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
    # 预训练word2vec
    :label:`sec_word2vec_pretraining`

    我们继续实现 :numref:`sec_word2vec`中定义的跳元语法模型。然后，我们将在PTB数据集上使用负采样预训练word2vec。首先，让我们通过调用`d2l.load_data_ptb`函数来获得该数据集的数据迭代器和词表，该函数在 :numref:`sec_word2vec_data`中进行了描述。
    """)
    return


@app.cell
def _():
    import math
    import torch
    from torch import nn
    from d2l import torch as d2l

    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                         num_noise_words)
    return d2l, data_iter, math, nn, torch, vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 跳元模型

    我们通过嵌入层和批量矩阵乘法实现了跳元模型。首先，让我们回顾一下嵌入层是如何工作的。

    ### 嵌入层

    如 :numref:`sec_seq2seq`中所述，嵌入层将词元的索引映射到其特征向量。该层的权重是一个矩阵，其行数等于字典大小（`input_dim`），列数等于每个标记的向量维数（`output_dim`）。在词嵌入模型训练之后，这个权重就是我们所需要的。
    """)
    return


@app.cell
def _(nn):
    embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
    print(f'Parameter embedding_weight ({embed.weight.shape}, '
          f'dtype={embed.weight.dtype})')
    return (embed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    嵌入层的输入是词元（词）的索引。对于任何词元索引$i$，其向量表示可以从嵌入层中的权重矩阵的第$i$行获得。由于向量维度（`output_dim`）被设置为4，因此当小批量词元索引的形状为（2，3）时，嵌入层返回具有形状（2，3，4）的向量。
    """)
    return


@app.cell
def _(embed, torch):
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    embed(x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 定义前向传播

    在前向传播中，跳元语法模型的输入包括形状为（批量大小，1）的中心词索引`center`和形状为（批量大小，`max_len`）的上下文与噪声词索引`contexts_and_negatives`，其中`max_len`在 :numref:`subsec_word2vec-minibatch-loading`中定义。这两个变量首先通过嵌入层从词元索引转换成向量，然后它们的批量矩阵相乘（在 :numref:`subsec_batch_dot`中描述）返回形状为（批量大小，1，`max_len`）的输出。输出中的每个元素是中心词向量和上下文或噪声词向量的点积。
    """)
    return


@app.cell
def _(torch):
    def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    return (skip_gram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们为一些样例输入打印此`skip_gram`函数的输出形状。
    """)
    return


@app.cell
def _(embed, skip_gram, torch):
    skip_gram(torch.ones((2, 1), dtype=torch.long),
              torch.ones((2, 4), dtype=torch.long), embed, embed).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 训练

    在训练带负采样的跳元模型之前，我们先定义它的损失函数。

    ### 二元交叉熵损失

    根据 :numref:`subsec_negative-sampling`中负采样损失函数的定义，我们将使用二元交叉熵损失。
    """)
    return


@app.cell
def _(nn):
    class SigmoidBCELoss(nn.Module):
        # 带掩码的二元交叉熵损失
        def __init__(self):
            super().__init__()

        def forward(self, inputs, target, mask=None):
            out = nn.functional.binary_cross_entropy_with_logits(
                inputs, target, weight=mask, reduction="none")
            return out.mean(dim=1)

    loss = SigmoidBCELoss()
    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    回想一下我们在 :numref:`subsec_word2vec-minibatch-loading`中对掩码变量和标签变量的描述。下面计算给定变量的二进制交叉熵损失。
    """)
    return


@app.cell
def _(loss, torch):
    pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
    label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    下面显示了如何使用二元交叉熵损失中的Sigmoid激活函数（以较低效率的方式）计算上述结果。我们可以将这两个输出视为两个规范化的损失，在非掩码预测上进行平均。
    """)
    return


@app.cell
def _(math):
    def sigmd(x):
        return -math.log(1 / (1 + math.exp(-x)))

    print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
    print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 初始化模型参数

    我们定义了两个嵌入层，将词表中的所有单词分别作为中心词和上下文词使用。字向量维度`embed_size`被设置为100。
    """)
    return


@app.cell
def _(nn, vocab):
    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    return (net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 定义训练阶段代码

    训练阶段代码实现定义如下。由于填充的存在，损失函数的计算与以前的训练函数略有不同。
    """)
    return


@app.cell
def _(d2l, loss, nn, skip_gram, torch):
    def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
        def init_weights(m):
            if type(m) == nn.Embedding:
                nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[1, num_epochs])
        # 规范化的损失之和，规范化的损失数
        metric = d2l.Accumulator(2)
        for epoch in range(num_epochs):
            timer, num_batches = d2l.Timer(), len(data_iter)
            for i, batch in enumerate(data_iter):
                optimizer.zero_grad()
                center, context_negative, mask, label = [
                    data.to(device) for data in batch]

                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                         / mask.sum(axis=1) * mask.shape[1])
                l.sum().backward()
                optimizer.step()
                metric.add(l.sum(), l.numel())
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}, '
              f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    现在，我们可以使用负采样来训练跳元模型。
    """)
    return


@app.cell
def _(data_iter, net, train):
    lr, num_epochs = 0.002, 5
    train(net, data_iter, lr, num_epochs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 应用词嵌入
    :label:`subsec_apply-word-embed`

    在训练word2vec模型之后，我们可以使用训练好模型中词向量的余弦相似度来从词表中找到与输入单词语义最相似的单词。
    """)
    return


@app.cell
def _(net, torch, vocab):
    def get_similar_tokens(query_token, k, embed):
        W = embed.weight.data
        x = W[vocab[query_token]]
        # 计算余弦相似性。增加1e-9以获得数值稳定性
        cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                          torch.sum(x * x) + 1e-9)
        topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
        for i in topk[1:]:  # 删除输入词
            print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

    get_similar_tokens('chip', 3, net[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    * 我们可以使用嵌入层和二元交叉熵损失来训练带负采样的跳元模型。
    * 词嵌入的应用包括基于词向量的余弦相似度为给定词找到语义相似的词。

    ## 练习

    1. 使用训练好的模型，找出其他输入词在语义上相似的词。您能通过调优超参数来改进结果吗？
    1. 当训练语料库很大时，在更新模型参数时，我们经常对当前小批量的*中心词*进行上下文词和噪声词的采样。换言之，同一中心词在不同的训练迭代轮数可以有不同的上下文词或噪声词。这种方法的好处是什么？尝试实现这种训练方法。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Discussions](https://discuss.d2l.ai/t/5740)
    """)
    return


if __name__ == "__main__":
    app.run()
