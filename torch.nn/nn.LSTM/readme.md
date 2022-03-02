# `nn.LSTM`

## `定义 LSTM 神经网络：`

    class LSTMNet(nn.Module):
        def __init__(self, vocab_size,embedding_dim, hidden_dim, layer_dim, output_dim):
            """
            vocab_size:词典长度
            embedding_dim:词向量的维度
            hidden_dim: RNN神经元个数
            layer_dim: RNN的层数
            output_dim:隐藏层输出的维度(分类的数量)
            """

            super(LSTMNet, self).__init__()
            self.hidden_dim = hidden_dim ## RNN神经元个数
            self.layer_dim = layer_dim ## RNN的层数
            
            ## 对文本进行词项量处理
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # LSTM ＋ 全连接层
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                                batch_first=True)
            
            self.fc1 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            embeds = self.embedding(x)
        
            # r_out shape (batch, time_step, output_size)
            # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
            # h_c shape (n_layers, batch, hidden_size)
            r_out, (h_n, h_c) = self.lstm(embeds, None)   # None 表示 hidden state 会用全0的 state
            
            # 选取最后一个时间点的out输出
            out = self.fc1(h_n[-1]) 
            
            return out