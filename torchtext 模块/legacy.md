# `torchtext.legacy`

## `一、介绍：`

* `torchtext.legacy` 中的 `data` 用于处理文本数据，从 `str` 转为 `index` ，再将数据换成`dataset` 格式。
 
        from torchtext.legacy import data

* `data` 中有三个模块，分别是：`Field、TabularDataset、BucketIterator`。


    * `Field` :`Torchtext` 采用声明式方法加载数据，需要先声明一个 `Field` 对象，这个Field对象指定你想要怎么处理某个数据，主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等等。
 
    * `TabularDataset` : `Field` 知道怎么处理原始数据，现在我们需要告诉 `Field` 去处理哪些数据。这就是我们需要用到 `Dataset` 的地方。 `TabularDataset` 继承自 `pytorch` 的 `Dataset` ，用于加载数据， `TabularDataset` 可以指点路径，格式， `Field` 信息就可以方便的完成数据加载。同时 `torchtext` 还提供预先构建的常用数据集的 `Dataset` 对象，可以直接加载使用， `splits` 方法可以同时加载训练集，验证集和测试集。

    * `BucketIterator` : 主要是数据输出的模型的迭代器，用于设置 `batch` ，相当于 `DataLoader` ，相比于标准迭代器，`BucketIterator` 会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，因此当样本长度差别较大时，使用 `BucketIerator` 可以带来填充效率的提高。除此之外，我们还可以在 `Field` 中通过 `fix_length` 参数来对样本进行截断补齐操作。。

## `二、参数说明：`

* `1、torchtext.data.Field：`

        torchtext.data.Field(sequential=True, use_vocab=True, init_token=None, eos_token=None, 
                             fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None, 
                             lower=False, tokenize=None, tokenizer_language='en',include_lengths=False, 
                             batch_first=False, pad_token='<pad>', unk_token='<unk>', 
                             pad_first=False, truncate_first=False, stop_words=None, is_target=False)

* `参数具体详解：`

        sequential: 是否把数据表示成序列，如果是False, 不能使用分词 默认值: True.

        use_vocab: 是否使用词典对象. 如果是False 数据的类型必须已经是数值类型. 默认值: True.

        init_token: 每一条数据的起始字符 默认值: None.

        eos_token: EOS 默认值: None.

        fix_length: 修改每条数据的长度为该值，不够的用pad_token补全. 默认值: None.

        tensor_type: 把数据转换成的tensor类型 默认值: torch.LongTensor.

        preprocessing:在分词之后和数值化之前使用的管道 默认值: None.

        postprocessing: 数值化之后和转化成tensor之前使用的管道默认值: None.

        lower: 是否把数据转化为小写 默认值: False.

        tokenize: 分词函数. 默认值: str.split.

        include_lengths: 是否返回一个已经补全的最小batch的元组和和一个包含每条数据长度的列表 . 默认值: False.

        batch_first: 是否Batch first. 默认值: False.

        pad_token: PAD 默认值: "".

        unk_token: UNK 默认值: "".

        pad_first: 是否补全第一个字符. 默认值: False.

* `2、torchtext.data.TranslationDataset`

        torchtext.datasets.TranslationDataset(path, exts, fields, **kwargs)

* `参数具体详解：`


        path: 两种语言的数据文件的路径的公共前缀

        exts: 包含每种语言路径扩展名的tuple

        fields: 包含将用于每种语言的Field的tuple

        **kwargs: 等等







## `三、用法：`

* `导入模块：`

        from torchtext.legacy import data

* `处理数据：Field`

        ## 使用torchtext库进行数据准备
        ## 定义文件中对文本和标签所要做的操作
        """
        sequential=True:表明输入的文本时字符，而不是数值字
        tokenize="mytokenize":使用 mytokenize 切分词语
        use_vocab=True: 创建一个词汇表
        batch_first=True: batch优先的数据方式
        fix_length=400 :每个句子固定长度为400，不足长度会自动用<pad>补齐
        """

        ## 定义文本切分方法，可以先用 jieba 对数据进行分词，然后这里直接设置空格切分即可
        mytokenize = lambda x: x.split()

        ## 定义文本处理的规则
        TEXT = data.Field(sequential=True, tokenize=mytokenize,   
                        include_lengths=True, use_vocab=True,
                        batch_first=True, fix_length=400)

        ## 定义标签的处理规则
        LABEL = data.Field(sequential=False, use_vocab=False,   
                        pad_token=None, unk_token=None)

        ## 对所要读取的数据集的列进行处理
        text_data_fields = [
            ("labelcode", LABEL), # 对标签的操作
            ("cutword", TEXT) # 对文本的操作
        ]

* `处理数据：TabularDataset`

        ## 使用 .splits 方法可以为多个数据集直接创建 Dataset
        ## 读取数据，TabularDataset 是 torchtext 内置的 Dataset 子类，能够方便的读取 csv。
        ## 不过有的时候需要自定义 dataset，就不能用它了。
        traindata,valdata,testdata = data.TabularDataset.splits(
            path="data/", format="csv", 
            train="cnews_train2.csv", fields=text_data_fields, 
            validation="cnews_val2.csv",
            test = "cnews_test2.csv", skip_header=True
        )

* `处理数据：BucketIterator`


        ## 定义一个迭代器，将类似长度的示例一起批处理。设置 batch,并且进行序列的补齐。
        BATCH_SIZE = 64
        train_iter = data.BucketIterator(traindata,batch_size = BATCH_SIZE)
        val_iter = data.BucketIterator(valdata,batch_size = BATCH_SIZE)
        test_iter = data.BucketIterator(testdata,batch_size = BATCH_SIZE)





## `参考：`


* `[TorchText]使用：`https://www.jianshu.com/p/e5adb235399e