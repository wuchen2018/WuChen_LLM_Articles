之前我以T5模型为例，演示如何debug 大语言模型。T5模型是 encoder decoder模型。GLM是decode only 模型，但是GLM的结构加入了一些比较独特的元素。主流的说法是，BERT的文本生成能力弱，GPT的上下文理解弱。GLM采用span完形填空，既产生了上下文能力，又产生了生成能力。那么GLM的完形填空到底是啥？它所谓的独特的mask机制是什么？我想通过debug的方式来弄明白。

首先，需要决定是debug训练阶段的模型还是推理阶段的模型？这个问题我一开始不清楚。由于debug推理模型更简单，因此我选择先从推理模型开始debug。

# 先找到源码在哪里
方法和T5一样，先投石问路，运行一下，看看是哪里报错。
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/home/notebook/data/group/huggingface/ChatGLM-models-dirs/chatglm3-6b", trust_remote_code=True)
model  = AutoModel.from_pretrained('/home/notebook/data/group/huggingface/ChatGLM-models-dirs/chatglm3-6b', trust_remote_code=True).half().cuda()
model('苹果的英文是什么？')
```
报错显示的路径是~/.cache/huggingface/modules/transformers_modules/chatglm3-6b/modeling_chatglm.py:937。可以看到这个文件并不像T5那样，位于transformers库的官方源代码里，而是自己的cache文件夹里。因为glm的部分处理代码并没有被收录在官方的代码里。参数加上trust_remote_code=True也正是这个原因。

# 找到源头（如何分词）
上一步找到源码的大概位置，目的是打一个断点。至于原始文本'苹果的英文是什么？'是怎么转成input_ids然后输入进模型的，我需要重新debug一次。重新debug的话，我需要改一下代码，不要再用model('苹果的英文是什么？')了，而是用model.chat，因为model.chat会调用整个从预处理，到分词，到推理，到后处理的所有流程。

使用这行代码：

response, history = model.chat(tokenizer, "苹果的英文是什么？", history=[])

然后debug，发现进入了断点（modeling_chatglm.py:937行）。断点的部分并不是开始，这个地方已经有input_ids了，至于这个input_ids是怎么来的，还要往前看。

这个时候，借用vscode的call stack的工具，可以一步一步往前看调用栈，看是从哪个函数开始进行预处理的。发现还是在modeling_chatglm.py文件里的chat函数里。然后顺藤摸瓜，发现逻辑在tokenization_chatglm.py函数里。

接下来看看具体的分词逻辑。

第一步，在query之前加入模板'<|user|>\n'，query之后加入'<|assistant|>'输入句子变成了：

'<|user|>\n苹果的英文是什么？<|assistant|>'

然后分词。（其实实际上是先对'<|user|>\n苹果的英文是什么？'分词，然后加上'<|assistant|>'的id，这样做似乎没有区别，可能是为了避免<|assistant|>和用户输入的语句混淆导致分词混乱）

第二步，配置停止词

eos_token是停止词。除此之外，还加了<|user|>和<|observation|>作为停止词。

# 模型的前向过程
'<|user|>\n苹果的英文是什么？<|assistant|>'经过分词后得到的input_ids是torch.Size([1, 12])的形状。

接下来进入embedding层，结果是torch.Size([12, 1, 4096])的形状。这倒是和一般的LLM不一样。一般的LLM的形状是torch.Size([1,12,4096])。（在modeling_chatglm.py726行，执行了[b s h] --> [s b h]的转换）为什么要这样转，我暂时也不清楚。

后面的步骤和一般的LLM都差不多了。

经过这一通debug，发现收获不多。只知道输入的模板是：

'<|user|>\n苹果的英文是什么？<|assistant|>'

至于所谓的MASK机制到底是啥？似乎都不知道。看来推理阶段的debug无法知道这些信息。还是得依靠训练阶段的debug。

# 借助xtuner的训练机制
我在之前分享过xtuner这个工具，这个工具可以方便地对主流的LLM进行微调。详情见我之前的分享：IntermLM（书生浦语模型）与基于Xtuner的微调实战 - 武辰的文章 - 知乎 https://zhuanlan.zhihu.com/p/681809497

微调命令是：xtuner train chatglm3_6b_qlora_alpaca_enzh_oasst1_e3_copy.py

在vscode上微调，也比较好配置：

{
    "name": "Python2: Xtuner Training",
    "type": "python",
    "request": "launch",
    "cwd": "/home/notebook/code/personal/aigc/intern/xtuner-main/ft-king20240319-chatglm",
    "program": "/opt/conda/bin/xtuner",
    "args": ["train", "chatglm3_6b_qlora_oasst1_e3_copy.py"],
    "console": "integratedTerminal",
    "justMyCode": false,
},
当我开始debug后，我发现竟然进不去modeling_chatglm.py，这真是一件奇怪的事情。没办法，只好从头开始debug。不过好在之前的文章有过一些记录。我可以直接在/opt/conda/lib/python3.10/site-packages/mmengine/runner/runner.py的build_dataloader函数打断点。

但是后面的debug之路并没有那么顺利。经过辛苦的探索，发现在源代码xtuner/xtuner/dataset/huggingface.py中负责加载数据并且做预处理。

第一个关键点在huggingface.py中的add_template_to_dataset

函数中的这一行代码：

dataset = dataset.map(template_map_fn, num_proc=map_num_proc)

我在之前的文章中也说过，这行代码的作用是加入对话模板。这一行代码比较费时间。

经过这个步骤，还没有分词。

第二个关键点在于tokenize_dataset函数中的分词。

经过分词后，dataset有两个字段，分别是input_ids和labels.

以我的数据集的第一条数据举例，第一条数据的input_ids和labels长度都是1021.

input_ids经过decode之后是：

'[gMASK]sop <|user|> \n {prompt} <|assistant|>\n {output}\n

output是：

[-100*966，30910，output(len=51)，2, -100, -100]

-100*966代表前面有966个-100，30910和2对应的字符都是''

注意，在input_ids的最后54个值，是[output(len=51)，2, 30910, 13]

13对应的字符是'\n'

至少input_ids和labels中对于output的部分，位置是对应的。（大部分LLM模型都是这样的）

经过这个分词，已经分好词了。这个步骤还有一个需要注意的点是[gMASK]sop 。它是如何加进去的，我debug了好久，就是进不去。但是在下载下来的文件tokenization_chatglm.py里面是有关于[gMASK]的信息。经过查阅资料：可以自由设置，单词是mask,句子是smask,文章是gmask,可以根据任务的不同设置mask，文本理解设置单词级别mask，文本生成色湖之句子级别的gmask，glm130B中设置的是70%句子级别gmask，30%词级别mask。

但是还没有生成attention mask之类的东西。只有inputs_ids和labels

第三个关键点是DataLoader（在runner.py）

生成了attention_mask，但是shape仍然是一维的，有些奇怪

input_ids和labels应该是没有变的，至少对应的output的位置没有变

我是很奇怪，为什么attention_mask是一维的，而且全部都是1？不应该是比较复杂的mask吗？

还有一个超级奇怪的问题，为什么debug不进去模型的前向过程？

现在我的关键应该是要debug进去模型的forward方法，但是进不去。（我遇到一件非常奇怪的事情。我在使用vscode调试python代码。这个代码是训练深度学习模型。我设置了"justMyCode": false。当我debug的时候，它可以正常地在不同的文件之间跳转。但是唯独在一个定义网络结构的python文件中，我在里面打了断点，仍然无法进去。奇怪的是，我在文件里添加了print函数，print函数是可以正常输出的，说明程序一定经过了这个地方。但是我在这个地方打断点，进不去。）

而且不同于之前发现代码是在cache里运行，这次的代码似乎是在模型的下载路径下运行。即，模型下载下来后，有一个modelling_chatglm.py文件，还有一个tokenization_chatglm.py文件。之前发现在cache文件夹里也有一样的文件，而且之前debug的时候发现它是在cache里的文件运行的。这一次，我在原文件写print(2)，在cache文件里写print(1)，发现打印出来的是2.然后我在print(2)处打断点，进不去。

可能还有一个可能性是，程序自动在哪个地方复制了一份代码，在新的地方运行，这或许可以回答我的疑问……

这个问题我至今没有解决。

对于attention_mask的问题，后来我才知道，在ChatGLM第二代，原本的复杂的mask结构，就已经被替换成了通用的casual mask，不再区分partA和partB了。而且二维位置编码也被ROPE取代了。

chatglm最独特的两个特征——二维位置编码和prefix mask，在第二代被彻底抛弃。

参考：

万字长文带你了解ChatGLM系列 - 暗影智芯的文章 - 知乎

https://zhuanlan.zhihu.com/p/696394009

https://www.bilibili.com/read/cv265
