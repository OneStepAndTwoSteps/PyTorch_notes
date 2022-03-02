# `torchtext.legacy`

## `一、介绍：`

* `torchtext.legacy` 中的 `data` 用于处理文本数据，从 `str` 转为 `index` ，再将数据换成`dataset` 格式。
 
        from torchtext.legacy import data

* `data` 中有三个模块，分别是：`Field、TabularDataset、BucketIterator`。


    * `Field` :主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等等。

    * `TabularDataset` :继承自 `pytorch` 的 `Dataset` ，用于加载数据， `TabularDataset` 可以指点路径，格式， `Field` 信息就可以方便的完成数据加载。同时 `torchtext` 还提供预先构建的常用数据集的 `Dataset` 对象，可以直接加载使用， `splits` 方法可以同时加载训练集，验证集和测试集。

    * `BucketIterator` : 主要是数据输出的模型的迭代器，用于设置 `batch` ，相当于 `DataLoader` ，相比于标准迭代器，`BucketIterator` 会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，因此当样本长度差别较大时，使用 `BucketIerator` 可以带来填充效率的提高。除此之外，我们还可以在 `Field` 中通过 `fix_length` 参数来对样本进行截断补齐操作。。


## `二、用法：`

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