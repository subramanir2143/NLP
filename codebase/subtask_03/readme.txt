####  INSTRUCTIONS FOR RUNNING THE FINAL IMPLEMENTATION   ####

1) Make sure the file 'task_3_event_prediction.tsv' is available in the location mentioned in cell number 6.

2) Make sure runtime is GPU. Execute the cells one by one. Cell number 13 will take around 15 minutes to complete.

3) Cell number 18 contains the train:test split ratio of 0.80. 

4) The two GRU models are defined in cells 20 and 21. Cell number 23 has the configuration for changing lead time(global_n_future).

5) Run the rest of the cells to execute the code for Dual-GRU network.

Lead time can be changed by simply changing the value stored in global_n_future variable.

Unidirectional RNN - Unidirectional RNN network:
Change the model name from GRU to RNN in cells 20 and 21 and re run to get the results for Unidir RNN - Unidir RNN network.

Unidirectional LSTM - Unidirectional LSTM network:
Change the model name from GRU to LSTM in cells 20 and 21. Also change the output of the model from output_1, h_n_1 => output_1, (h_n_1, c_n_1) and output_2, h_n_2 => output_2, (h_n_2, c_n_2)

Bidirectional GRU - Unidirectional GRU
Change the parameter(directions_1 to 2 and bidirectional_1 to True) passed to first GRU network. 
Replace the code in forward function of GRUClassifier1 with:

def forward(self, x, x_lens):        
        ###############################################################
        x_1 = x
        batch_1 = x_1.shape[0]
        # x_1: [batch, seq len, emb dim] => [5 * 80 * 810]

        # GRU 1
        x_1 = x_1.permute(1, 0, 2) # [seq len, batch, emb dim] ==> [80 * 5 * 810]

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(x_1, x_lens.to('cpu'), batch_first=False, enforce_sorted=False)        
        
        output_1, h_n_1 = self.gru_1(packed)
        # h_n_1 = self.dropout_1(h_n_1[-1, :, :])

        h_n_1 = h_n_1.view(-1, self.directions_1, batch_1, self.hidden_dim_1) # num_layers, directions, batch, hidden_size
        h_n_1 = h_n_1.sum(0) # directions, batch, hidden_size
        h_n_1 = torch.tanh(self.dropout_1(torch.cat([h_n_1[0,:,:], h_n_1[1,:,:]], -1))) # batch, 2*hidden_size


        # output_1 => [seq len * batch * hidden dim] ... [80 * 5 * 256]
        # h_n_1 => [num layers * batch * hidden dim] ... [2 * 5 * 256]
        # print(f'output dim 1: {output_1.shape}, h_n dim 1: {h_n_1.shape}')

        return h_n_1
