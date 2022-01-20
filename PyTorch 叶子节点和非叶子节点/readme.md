# `叶子节点和非叶子节点：`

* 在 `PyTorch` 的计算图`（computation graph）`中，只有 `叶子结点（leaf nodes）会保留梯度` 。而所有 `中间节点` 的梯度只被用于反向传播，一旦完成反向传播，中间节点的梯度就将自动释放，从而节约内存。

## `非叶子节点：`

* <font color= #EC7063  >__中间节点（并非直接指定数值的节点，而是由别的节点计算得到的节点）__</font > ，它们虽然 `requires_grad` 的参数都是 `True` ，但是反向传播后，它们的梯度并没有保存下来，而是直接删除了，因此是 `None` 。

* <font color= #EC7063  >__判断一个张量是不是叶子节点,可以通过它的属性 is_leaf 来查看。__</font > 

    `如下所示，`计算图中，`x y w` 为叶子节点，而 `z` 为中间节点：

        import torch

        x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
        y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
        w = torch.Tensor([1, 2, 3, 4]).requires_grad_()
        z = x+y
        # z.retain_grad()

        o = w.matmul(z)
        # o.retain_grad()

        o.backward()

        print('x.requires_grad:', x.requires_grad) # True
        print('y.requires_grad:', y.requires_grad) # True
        print('z.requires_grad:', z.requires_grad) # True
        print('w.requires_grad:', w.requires_grad) # True
        print('o.requires_grad:', o.requires_grad) # True


        print('x.grad:', x.grad) # tensor([1., 2., 3., 4.])
        print('y.grad:', y.grad) # tensor([1., 2., 3., 4.])
        print('w.grad:', w.grad) # tensor([ 4., 6., 8., 10.])
        print('z.grad:', z.grad) # None
        print('o.grad:', o.grad) # None

        


    `如果想在反向传播之后保留它们的梯度，则需要特殊指定：`把上面代码中的`z.retain_grad()` 和 `o.retain_grad` 的注释去掉，可以得到它们对应的梯度，运行结果如下所示：


        x.requires_grad: True
        y.requires_grad: True
        z.requires_grad: True
        w.requires_grad: True
        o.requires_grad: True
        x.grad: tensor([1., 2., 3., 4.])
        y.grad: tensor([1., 2., 3., 4.])
        w.grad: tensor([ 4.,  6.,  8., 10.])
        z.grad: tensor([1., 2., 3., 4.])
        o.grad: tensor(1.)


