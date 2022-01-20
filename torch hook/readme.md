# `torch 中的 hook`

 为了节省显存（内存），`pytorch` 在计算过程中只有叶子结点（`leaf nodes`）的节点会保留梯度,保存到该叶子节点张量的属性 `grad` 中，而所有中间节点的梯度只被用于反向传播，一旦完成反向传播，中间节点的梯度就将自动释放。

有时对网络进行分析时需要查看或修改这些中间节点，此时就需要注册一个钩子（`hook`）来导出需要的中间节点。

如果不设置 `hook` 的话也可以通过对中间节点设置 `.retain_grad()` 方法。

### `案例：register_forward_hook`

* 如果随机的初始化每个层，那么就无法测试出自己获取的输入输出是不是 `forward` 中的输入输出了，所以需要将每一层的权重和偏置设置为可识别的值（比如全部初始化为1）。网络包含两层（ `Linear` 有需要求导的参数被称为一个层，而 `ReLU` 没有需要求导的参数不被称作一层），`__init__()` 中调用 `initialize` 函数对所有层进行初始化。


        import torch
        import torch.nn as nn

        class TestForHook(nn.Module):
            def __init__(self):
                super().__init__()

                self.linear_1 = nn.Linear(in_features=2, out_features=2)
                self.linear_2 = nn.Linear(in_features=2, out_features=1)
                self.relu = nn.ReLU()
                self.relu6 = nn.ReLU6()
                self.initialize()

            def forward(self, x):
                linear_1 = self.linear_1(x)
                linear_2 = self.linear_2(linear_1)
                relu = self.relu(linear_2)
                relu_6 = self.relu6(relu)
                layers_in = (x, linear_1, linear_2)
                layers_out = (linear_1, linear_2, relu)
                return relu_6, layers_in, layers_out

            def initialize(self):
                """ 定义特殊的初始化，用于验证是不是获取了权重"""
                self.linear_1.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
                self.linear_1.bias = torch.nn.Parameter(torch.FloatTensor([1, 1]))
                self.linear_2.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
                self.linear_2.bias = torch.nn.Parameter(torch.FloatTensor([1]))
                return True


        # 1：定义用于获取网络各层输入输出tensor的容器
        # 并定义module_name用于记录相应的module名字
        module_name = []
        features_in_hook = []               ## 用于保存 hook 得到的 fea_in
        features_out_hook = []              ## 用于保存 hook 得到的 fea_out


        # 2：hook函数负责将获取的输入输出添加到feature列表中
        # hook函数需要三个参数，这三个参数是系统传给hook函数的，自己不能修改这三个参数：
        # 并提供相应的module名字
        # features_in_hook 储存了注册的hook层中的输入特征
        # features_out_hook 储存了注册的hook层中的输出特征
        def hook(module, fea_in, fea_out):
            print("hooker working")
            print('module.__class__：',module.__class__)
            module_name.append(module.__class__)
            print('fea_in：',fea_in)
            print('fea_out：',fea_out)
            print('\n')
            features_in_hook.append(fea_in)
            features_out_hook.append(fea_out)
            return None


        # 3：定义全部是1的输入
        x = torch.FloatTensor([[0.1, 0.1], [0.1, 0.1]])


        # 4:注册钩子可以对某些层单独进行,以下对不是 nn.ReLU6 的层全部注册了hook.
        # 注册钩子必须在forward（）函数被执行之前，也就是定义网络进行计算之前就要注册.
        # 以后只要进行前项传播操作，hook 函数都会将 fea_in 和 fea_out 保存到 features_in_hook 和 features_out_hook 中
        net = TestForHook()
        net_chilren = net.children()
        for child in net_chilren:
            if not isinstance(child, nn.ReLU6):
                child.register_forward_hook(hook=hook)
        ## list(net_chilren)
        ## Linear(in_features=2, out_features=2, bias=True)
        ## Linear(in_features=2, out_features=1, bias=True)
        ## ReLU()
        ## ReLU6()


        # 5:测试网络输出
        out, features_in_forward, features_out_forward = net(x)
        print("*"*5+"forward return features"+"*"*5)
        print('features_in_forward：',features_in_forward)
        print('features_out_forward：',features_out_forward)
        print("*"*5+"forward return features"+"*"*5)


        # 6:测试features_in是不是存储了输入
        ## 在上面的代码中指定了前三层的hook，features_in_hook 是该层的输入特征，features_out_hook 是该层的输出特征。
        ## 可以看到进行forward之后，最后返回的所有层的forward值和hook层勾出来的值是一样的。

        print("*"*5+"hook record features"+"*"*5)
        print(features_in_hook)
        print(features_out_hook)
        print(module_name)
        print("*"*5+"hook record features"+"*"*5)


        # 7：测试forward返回的feautes_in是不是和hook记录的一致(通过减法来查看)
        print("sub result")
        for forward_return, hook_record in zip(features_in_forward, features_in_hook):
            print(forward_return-hook_record[0])

