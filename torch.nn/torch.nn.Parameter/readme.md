# `torch.nn.Parameter`

## `简介：`

* `torch.nn.Parameter` 是继承自 `torch.Tensor` 的子类，其主要作用是作为 `nn.Module` 中的可训练参数使用。它与 `torch.Tensor` 的区别就是 `nn.Parameter` 会自动被认为是 `module` 的可训练参数，即加入到 `parameter()` 这个迭代器中去；而 `module` 中非 `nn.Parameter()` 的普通 `tensor` 是不在 `parameter` 中的。


## `参考链接：`

* `PyTorch里面的torch.nn.Parameter()：`https://www.jianshu.com/p/d8b77cc02410


* `torch.nn.Parameter理解：`https://blog.csdn.net/qq_28753373/article/details/104179354