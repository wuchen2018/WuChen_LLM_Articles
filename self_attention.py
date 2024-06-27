import torch 


class SelfAttention():
    def __init__(self,layer_idx):
        self.q_proj = torch.nn.Linear(hidden_size, num_heads*head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads*head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads*head_dim)
        self.o_proj = torch.nn.Linear(num_heads*head_dim, hidden_size)
        self.layer_idx = layer_idx
        
    def apply_rotary_pos_emb(self,q,k):
        #just for demonstration
        return q,k
    
    def scaled_dot_product_attention(self,query,key,value,attention_mask):
        #qkv的shape=bs*16*length*128
        scale = query.shape[-1]**0.5 #128**0.5
        attn_weight = query @ key.transpose(-2, -1) / scale #shape=bs*16*length*length
        attn_weight += attention_mask #attention_mask是下三角矩阵。右上角都是-inf，左下角都是0
        attn_weight = torch.softmax(attn_weight,dim=-1)
        return attn_weight @ value
        
    def forward(self,hidden_states,attention_mask,past_key_value):
        #hidden_states是上一步的输出，在第一次推理生成的时候，shape是bs*length*2048
        #由于kvcache的存在，从第二次推理开始，只需要将一个token输入进网络中，对应的shape=bs*1*2048，就是length=1的情况
        #第一步：三分天下，获得qkv
        query = self.q_proj(hidden_states) #1*length*2048
        key = self.k_proj(hidden_states) #1*length*2048
        value = self.v_proj(hidden_states) #1*length*2048
        
        #第二步：转成多头注意力形状
        query = query.view(bs,seq_length,num_heads,head_dim).transpose(1,2) #torch.Size([1, 16, length, 128])
        key = key.view(bs,seq_length,num_heads,head_dim).transpose(1,2)
        value = value.view(bs,seq_length,num_heads,head_dim).transpose(1,2)
        
        #第三步：应用相对位置编码
        query, key = self.apply_rotary_pos_emb(query, key) #shape不会改变，1*16*length*128
        
        #第四步：更新kvcache（注意是在应用完相对位置编码之后更新的）        
        # query,key=past_key_value.update(query,key) 
        '''
        假设截止到此时句子的总长度是length。在这个步骤之前，
        query,key的shape是bs*16*1*128，past_key_value的shape是[#2][#layer_nums](#bs*16*(length-1)*128)
        在这个步骤中，将此步骤的query和key更新到past_key_value中，
        更新之后，past_key_value的shape是[#2][#layer_nums](#bs*16*length*128)
        并且将past_key_value中的query和key抽出来，变成当前的query和key
        query,key的shape变成了bs*16*length*128，和value是一样的shape了
        '''
        
        #第五步：点积缩放注意力
        attn_output = self.scaled_dot_product_attention(query,key,value,attention_mask)#([1, 16, 6, 128])
        
        #第六步：从多头注意力形状转回来
        attn_output = attn_output.transpose(1,2).reshape(bs,seq_length,hidden_size) #1*length*2048
        
        #第七步：输出投影层
        attn_output = self.o_proj(attn_output) #1*length*2048
        
        return attn_output,past_key_value
bs = 4
seq_length = 6
hidden_size = 2048
num_heads = 16
head_dim = 128
hidden_states = torch.randn(bs,seq_length,hidden_size) #4*6*2048
attention_mask = torch.randn(bs,num_heads,seq_length,seq_length) 
past_key_value = None

s = SelfAttention(0)
re = s.forward(hidden_states,attention_mask,past_key_value)
        
