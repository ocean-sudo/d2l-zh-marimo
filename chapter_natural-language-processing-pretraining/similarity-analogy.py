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
    # 词的相似性和类比任务
    :label:`sec_synonyms`

    在 :numref:`sec_word2vec_pretraining`中，我们在一个小的数据集上训练了一个word2vec模型，并使用它为一个输入词寻找语义相似的词。实际上，在大型语料库上预先训练的词向量可以应用于下游的自然语言处理任务，这将在后面的 :numref:`chap_nlp_app`中讨论。为了直观地演示大型语料库中预训练词向量的语义，让我们将预训练词向量应用到词的相似性和类比任务中。
    """)
    return


@app.cell
def _():
    import os
    import torch
    from torch import nn
    from d2l import torch as d2l

    return d2l, os, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 加载预训练词向量

    以下列出维度为50、100和300的预训练GloVe嵌入，可从[GloVe网站](https://nlp.stanford.edu/projects/glove/)下载。预训练的fastText嵌入有多种语言。这里我们使用可以从[fastText网站](https://fasttext.cc/)下载300维度的英文版本（“wiki.en”）。
    """)
    return


@app.cell
def _(d2l):
    #@save
    d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                    '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

    #@save
    d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                     'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

    #@save
    d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                      'b5116e234e9eb9076672cfeabf5469f3eec904fa')

    #@save
    d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                               'c1816da3821ae9f43899be655002f6c723e91b88')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    为了加载这些预训练的GloVe和fastText嵌入，我们定义了以下`TokenEmbedding`类。
    """)
    return


@app.cell
def _(d2l, os, torch):
    #@save
    class TokenEmbedding:
        """GloVe嵌入"""
        def __init__(self, embedding_name):
            self.idx_to_token, self.idx_to_vec = self._load_embedding(
                embedding_name)
            self.unknown_idx = 0
            self.token_to_idx = {token: idx for idx, token in
                                 enumerate(self.idx_to_token)}

        def _load_embedding(self, embedding_name):
            idx_to_token, idx_to_vec = ['<unk>'], []
            data_dir = d2l.download_extract(embedding_name)
            # GloVe网站：https://nlp.stanford.edu/projects/glove/
            # fastText网站：https://fasttext.cc/
            with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
                for line in f:
                    elems = line.rstrip().split(' ')
                    token, elems = elems[0], [float(elem) for elem in elems[1:]]
                    # 跳过标题信息，例如fastText中的首行
                    if len(elems) > 1:
                        idx_to_token.append(token)
                        idx_to_vec.append(elems)
            idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
            return idx_to_token, torch.tensor(idx_to_vec)

        def __getitem__(self, tokens):
            indices = [self.token_to_idx.get(token, self.unknown_idx)
                       for token in tokens]
            vecs = self.idx_to_vec[torch.tensor(indices)]
            return vecs

        def __len__(self):
            return len(self.idx_to_token)

    return (TokenEmbedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    下面我们加载50维GloVe嵌入（在维基百科的子集上预训练）。创建`TokenEmbedding`实例时，如果尚未下载指定的嵌入文件，则必须下载该文件。
    """)
    return


@app.cell
def _(TokenEmbedding):
    glove_6b50d = TokenEmbedding('glove.6b.50d')
    return (glove_6b50d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    输出词表大小。词表包含400000个词（词元）和一个特殊的未知词元。
    """)
    return


@app.cell
def _(glove_6b50d):
    len(glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们可以得到词表中一个单词的索引，反之亦然。
    """)
    return


@app.cell
def _(glove_6b50d):
    glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 应用预训练词向量

    使用加载的GloVe向量，我们将通过下面的词相似性和类比任务中来展示词向量的语义。

    ### 词相似度

    与 :numref:`subsec_apply-word-embed`类似，为了根据词向量之间的余弦相似性为输入词查找语义相似的词，我们实现了以下`knn`（$k$近邻）函数。
    """)
    return


@app.cell
def _(torch):
    def knn(W, x, k):
        # 增加1e-9以获得数值稳定性
        cos = torch.mv(W, x.reshape(-1,)) / (
            torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
            torch.sqrt((x * x).sum()))
        _, topk = torch.topk(cos, k=k)
        return topk, [cos[int(i)] for i in topk]

    return (knn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    然后，我们使用`TokenEmbedding`的实例`embed`中预训练好的词向量来搜索相似的词。
    """)
    return


@app.cell
def _(knn):
    def get_similar_tokens(query_token, k, embed):
        topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
        for i, c in zip(topk[1:], cos[1:]):  # 排除输入词
            print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')

    return (get_similar_tokens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `glove_6b50d`中预训练词向量的词表包含400000个词和一个特殊的未知词元。排除输入词和未知词元后，我们在词表中找到与“chip”一词语义最相似的三个词。
    """)
    return


@app.cell
def _(get_similar_tokens, glove_6b50d):
    get_similar_tokens('chip', 3, glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    下面输出与“baby”和“beautiful”相似的词。
    """)
    return


@app.cell
def _(get_similar_tokens, glove_6b50d):
    get_similar_tokens('baby', 3, glove_6b50d)
    return


@app.cell
def _(get_similar_tokens, glove_6b50d):
    get_similar_tokens('beautiful', 3, glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 词类比

    除了找到相似的词，我们还可以将词向量应用到词类比任务中。
    例如，“man” : “woman” :: “son” : “daughter”是一个词的类比。
    “man”是对“woman”的类比，“son”是对“daughter”的类比。
    具体来说，词类比任务可以定义为：
    对于单词类比$a : b :: c : d$，给出前三个词$a$、$b$和$c$，找到$d$。
    用$\text{vec}(w)$表示词$w$的向量，
    为了完成这个类比，我们将找到一个词，
    其向量与$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$的结果最相似。
    """)
    return


@app.cell
def _(knn):
    def get_analogy(token_a, token_b, token_c, embed):
        vecs = embed[[token_a, token_b, token_c]]
        x = vecs[1] - vecs[0] + vecs[2]
        topk, cos = knn(embed.idx_to_vec, x, 1)
        return embed.idx_to_token[int(topk[0])]  # 删除未知词

    return (get_analogy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们使用加载的词向量来验证“male-female”类比。
    """)
    return


@app.cell
def _(get_analogy, glove_6b50d):
    get_analogy('man', 'woman', 'son', glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    下面完成一个“首都-国家”的类比：
    “beijing” : “china” :: “tokyo” : “japan”。
    这说明了预训练词向量中的语义。
    """)
    return


@app.cell
def _(get_analogy, glove_6b50d):
    get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    另外，对于“bad” : “worst” :: “big” : “biggest”等“形容词-形容词最高级”的比喻，预训练词向量可以捕捉到句法信息。
    """)
    return


@app.cell
def _(get_analogy, glove_6b50d):
    get_analogy('bad', 'worst', 'big', glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    为了演示在预训练词向量中捕捉到的过去式概念，我们可以使用“现在式-过去式”的类比来测试句法：“do” : “did” :: “go” : “went”。
    """)
    return


@app.cell
def _(get_analogy, glove_6b50d):
    get_analogy('do', 'did', 'go', glove_6b50d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    * 在实践中，在大型语料库上预先练的词向量可以应用于下游的自然语言处理任务。
    * 预训练的词向量可以应用于词的相似性和类比任务。

    ## 练习

    1. 使用`TokenEmbedding('wiki.en')`测试fastText结果。
    1. 当词表非常大时，我们怎样才能更快地找到相似的词或完成一个词的类比呢？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Discussions](https://discuss.d2l.ai/t/5746)
    """)
    return


if __name__ == "__main__":
    app.run()
