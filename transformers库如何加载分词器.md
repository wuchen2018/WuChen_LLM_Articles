本文通过chatGLM的例子，来分析transformers库加载分词器的逻辑。

# 代码
```python
from tokenizer.tokenization_chatglm import ChatGLMTokenizer
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('./tokenizer',trust_remote_code=True)
# tokenizer = ChatGLMTokenizer(vocab_file='./tokenizer/tokenizer.model')


text_id=tokenizer.encode('“你好吗？”的英文翻译是：Are you OK?',
add_special_tokens=True)
```
其中"./tokenizer"是一个相对路径，该路径下包括三个文件：tokenization_chatglm.py、tokenizer_config.json、tokenizer.model

tokenization_chatglm.py： 这是一个Python脚本文件，通常包含ChatGLM模型专用的tokenizer类及其相关方法的实现。它定义了如何将原始文本（自然语言句子）转换为模型可以理解的token序列（整数索引列表）的过程。

tokenizer_config.json： 这是一个JSON格式的配置文件，包含了tokenizer的重要参数和设置。在本例子中的内容如下：
```python
{
  "name_or_path": "THUDM/chatglm2-6b",
  "remove_space": false,
  "do_lower_case": false,
  "tokenizer_class": "ChatGLMTokenizer",
  "auto_map": {
    "AutoTokenizer": [
      "tokenization_chatglm.ChatGLMTokenizer",
      null
      ]
  }
}
```
tokenizer_config.json文件为tokenizer提供了静态配置信息，使得在不直接查看源代码的情况下也能了解tokenizer的基本行为和特性。这些配置数据通常在加载tokenizer时被读取并用于初始化tokenizer对象。

tokenizer.model： 这个文件通常存储了tokenizer的模型状态或数据结构，具体取决于tokenizer的类型。对于基于子词的tokenizer（如BPE），这个文件通常包含：

1.词汇表：存储所有已学习到的子词及其对应的整数索引。

2.合并规则：记录了如何从基础字符逐步构建更复杂的子词的合并规则或概率信息。

# 通过AutoTokenizer加载分词器
这一行代码加载了分词器：
```python
tokenizer = AutoTokenizer.from_pretrained('./tokenizer',
trust_remote_code=True)
```
通过AutoTokenizer加载模型，使用的代码是：
```python
AutoTokenizer.from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
```
在这行代码中，pretrained_model_name_or_path参数有以下几种形式：

1.字符串形式：这是预定义分词器的模型id，这种分词器被托管在 http://huggingface.co 的模型库中。有效的模型id可以在根级别，比如 bert-base-uncased，也可以在用户或组织名之下，比如 dbmdz/bert-base-german-cased。

2.单个文件的路径或URL（只针对只需要一个词汇文件的分词器如Bert或XLNet）。举例来说，可以是./my_model_directory/vocab.txt这样的路径。

3.指向一个目录的路径：这个目录需要包含分词器所需的词汇文件，例如通过 PreTrainedTokenizer.save_pretrained 方法保存的文件，例如，./my_model_directory/。

在AutoTokenizer.from_pretrained的源代码中，程序会对pretrained_model_name_or_path做判断。如果是一个文件名，或者是一个url，则载入文件。然后，如果是一个文件夹，则载入该文件夹。如果以上都不是，则调用hf_hub_download从huggingface仓库里下载对应的分词器/数据/模型。

hf_hub_download 是一个用于从 Hugging Face Model Hub (http://hf.co/models) 或 Hugging Face Dataset Hub (http://hf.co/datasets) 下载模型或数据集文件的方法。它是 huggingface_hub 库提供的功能，该库是为方便访问和管理Hugging Face Hub上的资源而设计的。一个简单的例子是：
```python
from huggingface_hub import hf_hub_download
repo_id = "tiiuae/falcon-7b-instruct"
cache_dir = "/path/to/custom/cache"
hf_hub_download(repo_id, cache_dir=cache_dir)
```
经过AutoTokenizer.from_pretrained后，tokenizer已经被加载。打印tokenizer后，它的值等于：
```python
ChatGLMTokenizer(name_or_path='./tokenizer', vocab_size=64794, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='left', truncation_side='right', special_tokens={'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False), added_tokens_decoder={
}
```
这是是一个被调用的 tokenize 方法，它是 ChatGLMTokenizer 类的一个实例方法。这个方法主要有以下几个属性：

name_or_path : 分词器的名称或者路径，这里是’./tokenizer’。

vocab_size : 分词器的词汇表大小，这里是64794。

model_max_length : 分词器处理的最大文本长度，这里是一个非常大的数。在某些情况下，tokenizer的model_max_length属性可能被初始化为一个极大的数值作为默认值。这可能是为了确保在大多数使用场景下，tokenizer不会因为文本长度超出限制而引发错误。这种设定允许tokenizer处理极长的文本序列，但实践中很少会遇到如此长的输入。

is_fast : 是否使用fast tokenizer，这里是False。

padding_side , truncation_side : 分词器的填充和截断策略，分别是’left’和’right’。

special_tokens : 分词器的特殊符号，这里包括’eos_token’, ‘unk_token’ 和 ‘pad_token’。

clean_up_tokenization_spaces : 在分词后是否清理空格，这里是False。

added_tokens_decoder : 这是一个词典，存储了分词器中所有额外添加的标记及其对应的ID。

# 通过ChatGLMTokenizer加载分词器
除了通过AutoTokenizer.from_pretrained的方法，直接通过本地路径里的ChatGLMTokenizer类也能加载分词器：
```python
tokenizer = ChatGLMTokenizer(vocab_file='./tokenizer/tokenizer.model')
```
这两种方法是等价的。
