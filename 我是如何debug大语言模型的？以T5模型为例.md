有人私信我，问我如何debug大语言模型。我正好将去年做的一个笔记拿出来，和大家一起分享、讨论。
相信很多人学习大模型的路径和我一样，是直接从GPT这种decoder-only的模型或者BERT这种encoder-only的模型开始的。还有一种模型架构叫做encoder-decorder模型，但是一直没有过多去注意。其实2017年提出的Transformer网络本身就是encoder-decorder模型，BERT的研究人员单独把encoder模型拿出来，GPT的研究人员单独把decorder拿出来。还有一波人参照原始的transformer架构，发展了encoder-decorder模型，比如T5。
本文分享我debug T5模型代码的历程。
# 如何顺藤摸瓜找到源码？
我之前没有接触过encoder-decorder模型。在huggingface下载了一个T5模型后，首先参照网上的quick start跑一下，发现可以跑通，只是生成的结果有些差，但这不是当前的主要问题。当前的主要问题是，这个模型的前向推理过程是啥？这个模型的输入是啥？encoder-decorder模型到底是什么机制？
quick start代码：
```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


ds = Dataset.load_from_disk("/home/notebook/code/personal/aigc/trans/transformers-code/02-NLP Tasks/15-text_summarization/nlpcc_2017")
ds = ds.train_test_split(100, seed=42)
tokenizer = AutoTokenizer.from_pretrained('/home/notebook/data/group/huggingface/T5-models-dirs/mengzi-t5-base')


def process_func(exmaples):
    contents = ["摘要生成: \n" + e for e in exmaples["content"]]
    inputs = tokenizer(contents, max_length=384, truncation=True)
    labels = tokenizer(text_target=exmaples["title"], max_length=64, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs


tokenized_ds = ds.map(process_func, batched=True)
tokenized_ds


model  = AutoModelForSeq2SeqLM.from_pretrained('/home/notebook/data/group/huggingface/T5-models-dirs/mengzi-t5-base')


from transformers import pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
pipe("摘要生成:\n"+ds["test"][0]["content"], max_length=64, do_sample=True)
```
为了解决我的问题，我的办法是，先运行一下model('hello')，然后看报错，定位到了这一行的错误：
/opt/conda/envs/transformers2/lib/python3.9/site-packages/transformers/models/t5/modeling_t5.py:1711, in T5ForConditionalGeneration.forward(.............)
于是这就顺藤摸瓜找到了源代码。（经过简单的了解，发现T5ForConditionalGeneration这个类是T5模型中用于生成任务的类）。于是我在这附近几行代码里打了断点。
# 编码器
打了断点后，发现不太对劲，为什么在这个类中，模型的输入有一个encoder_outputs？一般来说，模型的输入是input_ids，是一些整数，怎么在这个类的forward方法，模型的输入看起来是已经经过了某种编码的张量？
于是利用vscode的CALL_STACK栏，我终于在/opt/conda/envs/transformers2/lib/python3.9/site-packages/transformers/generation/utils.py找到了这几行代码：
```python
if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs: 
    # if model is encoder decoder encoder_outputs are created
    # and added to `model_kwargs`
    model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(  #在这一步进行编码！
        inputs_tensor, model_kwargs, model_input_name
    )
```
这几行的代码的意思是，如果这个模型是encoder_decoder模型，那么先将输入运输到模型的encoder模块中去，编码器产生的结果整合到model_kwargs中。
进入这行代码单步调试，的确发现，input_ids被送进了模型的encoder模块中，产生了encoder_outputs.这里的input_ids的shape是1*310。经过编码后，model_kwargs["encoder_outputs"].last_hidden_state.shape是torch.Size([1, 310, 768])
# 解码器的输入是什么？
现在，模型的编码器已经将输入编码了。接下来就要考虑解码器了。编码器的输入是input_ids（可能还包括attention_mask，但是这个矩阵的结果都是1，可能是因为整个语料都视为一句话）。那么解码器的输入是什么呢？
解码器的输入包括两个（我是如何知道的？也是一行一行代码debug出来的，详见后文），第一个是解码器的input_ids，解码器的input_ids不是编码器的input_ids，而是：tensor([[0]], device='cuda:0')
这个张量是啥？我寻思着，看到有一行代码是：
decoder_start_token_id=generation_config.decoder_start_token_id
这行代码的意思是，解码器的初始token就是被设定为0.我调用tokenizer.decode后发现，对应的字符是<pad>。可能它的意思是想让模型输出的第一个token是<pad>。不管怎么说，这是模型的设定。这个设定其实可以在config.json文件里看到，里面有一decoder_start_token_id的字段。
解码器的第二个输入就是encoder_outputs.last_hidden_state，shape=torch.Size([1, 310, 768])，就是编码器的输出。
这是调用解码器的代码：
```python
decoder_outputs = self.decoder(
    input_ids=decoder_input_ids,
    attention_mask=decoder_attention_mask,
    inputs_embeds=decoder_inputs_embeds,
    past_key_values=past_key_values,
    encoder_hidden_states=hidden_states,
    encoder_attention_mask=attention_mask,
    head_mask=decoder_head_mask,
    cross_attn_head_mask=cross_attn_head_mask,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
```
一行一行去看上面代码的输入，在输入方面，只有decoder_input_ids/hidden_states是有值的，attention_mask全部是1，应该没啥用。decoder_input_ids就是[0]，hidden_states就是编码器的输出。
# 解码器里面有什么？
（源码：/opt/conda/envs/transformers2/lib/python3.9/site-packages/transformers/models/t5/modeling_t5.py中的T5Block）
接下来，来看看解码器里面有什么。（你可能会问为什么不看编码器里有什么，其实编码器就是普通的encoder，如果你熟悉了BERT就不必再去重复debug编码器里面的内容）
上面说到，编码器的输入有两个：[0]和编码器的输出。
[0]经过embedding层后，变成了一个768维的向量。
每一个解码器的block，主要包括三个步骤
第一个步骤是自注意力机制，只需把解码器的输入送进去即可。在这里，是把那个768维的向量送进去。（注意，这只是第一轮的情况。在第二轮，需要把2\*768的数据送进去。以此类推，直到模型生成的过程结束。具体参考transformer的生成机制）
得到1\*1\*768的输出。
第二个步骤是交叉注意力。
输入是上一个步骤的输出（1*1*768），以及编码器的输出（1*310*768），来做一个交叉注意力。交叉注意力，具体来说，在注意力机制中，Q来自解码器上一个步骤的输出，K和V来自编码器。
生成的结果仍然是1*1*768
第三个步骤是前馈层，比较简单，就不说了。
剩下的步骤和普通的decoder only类似了，就不再细说了。有一个细节是，虽然每次推理的时候，encoder都要参与，但是encoder只需要计算一次，在每次predict next word的时候，encoder都是不变的（有点像kvcache的感觉）。
