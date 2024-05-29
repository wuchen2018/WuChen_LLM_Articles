在之前的文章中，我分享了如何使用xtuner框架来微调IntermLM（书生浦语模型）：武辰：IntermLM（书生浦语模型）与基于Xtuner的微调实战

经过微调后，我们需要在测试集上对模型进行测试。尽管xtuner提供了chat接口，但这种接口主要适用于命令行交互，对于需要将结果输出到文件的场景并不太友好。另外，有时候并不适合使用专业的测试数据集去测试效果，而是需要主观测试。为了满足这些需求，这里我将介绍如何使用transformers库来实现模型推理。

我的需求是在100条数据的测试集上进行测试，模型的结果需要导出到CSV文件中。CSV文件的第一列是prompt（或者input），第二列是label，第三列是模型的输出。这种方式可以直观地展示模型的效果。

需要注意的是推理过程非常简单，一句代码即可完成：model.generate。但是在使用过程中还有一些细节需要注意。

# 细节一：加载对话模板
代码如下：
```python
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from xtuner.utils import PROMPT_TEMPATE, SYSTEM_TEMPLATE
from datasets import Dataset, load_dataset
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

# 载入预训练的模型
model_path = "##模型所在的path##"
tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="auto", trust_remote_code=True)

# 载入对话模板
prompt_template = PROMPT_TEMPLATE.internlm2_chat
print('===template===',prompt_template)

# 加载数据
path = "##数据所在的path##"
data = load_dataset("json", data_files=path+"test.json")
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto", trust_remote_code=True)
def template_map_fn(example):
    # print(1)
    conversation = example.get('conversation', [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation.get('input', '')
        if input is None:
            input = ''
        input_text = prompt_template['INSTRUCTION'].format(input=input, round=i + 1)
        system = single_turn_conversation.get('system', '')
        if system != '' and system is not None:
            system = prompt_template['SYSTEM'].format(system=system)
            input_text = system + input_text
        single_turn_conversation['input'] = input_text

        if prompt_template.get('SUFFIX', None):
            output_text = single_turn_conversation.get('output', '')
            output_text += prompt_template['SUFFIX']
            single_turn_conversation['output'] = output_text

        # SUFFIX_AS_EOS is False ==> need_eos_token is True
        single_turn_conversation['need_eos_token'] = \
            not prompt_template.get('SUFFIX_AS_EOS', False)
        single_turn_conversation['sep'] = prompt_template.get('SEP', '')

    return {'conversation': conversation}
data = data.map(template_map_fn, num_proc=32)  # `num_proc`是可选的，并行处理加快速度
```
注意，在下面这行代码下，我载入了对话模板：
```python
prompt_template = PROMPT_TEMPLATE.internlm2_chat
```
为什么要载入对话模板？要训练出大语言对话模型，模型的数据输入需要遵循一定的格式，这样模型才能够区分出哪些是用户说的话，哪些是模型回复的话。训练过程中的所有数据都遵从一种特定的对话模板。因此，当我们进行推理时，也需要正确地配对相应的对话模板。

在我的代码中，我使用了xtuner库，如果你不方便安装这个库，可以直接在xtuner的github页面中（https://github.com/InternLM/xtuner/blob/main/xtuner/utils/templates.py ）上找到你需要的模型模板，然后复制到你的代码中。这样，你就可以对你的对话模型进行正确的推理操作了。

比如，这是我在xtuner源码中找到的internlm2_chat模型的template：
```python
    internlm2_chat=dict(
        SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>']),
```

# 细节二：配置停止词
如果在调用model.generate时没有指定stopping_criteria参数，生成器将使用默认的停止条件。默认情况下，生成器的停止条件有：

1.达到最大生成长度：通过max_length参数设置，默认值通常为20。生成器在达到这个长度时将停止生成。

2.达到结束（end of sequence）符号：许多预训练语言模型都使用特殊的结束符号（通常是[EOS]或[SEP]）。当生成器生成了这个结束token时，生成过程就会停止。这些token在不同的模型中对应的ID可能有所不同，可以在tokenizer.eos_token_id中找到模型对应的结束符号ID。在internlm2-chat模型中，tokenizer.eos_token是'</s>'。

这些默认条件可以确保生成文本不会无休止地持续产生。然而，在某些情况下，你可能需要按照特定的任务需求自定义停止条件。为此，可以通过在调用generate函数时添加stopping_criteria参数来添加自定义停止条件。

代码：
```python
# 自定义停止词
class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

stop_criteria = StoppingCriteriaList()
stop_words = prompt_template['STOP_WORDS'].copy()
for word in stop_words:
    stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
```
在这段代码中，我们将prompt_template[‘STOP_WORDS’]中的STOP_WORDS加入到了生成过程的停止条件。以前面提到的internlm2-chat模型为例，它的停止词是'<|im_e0nd|>'。

这段代码的逻辑在于在生成每一个新的token之后，都会将当前文本的最后一部分（与停止词长度相同）与停止词进行比较，即判断cur_text[-self.length:] == self.stop_word是否成立。

举例来说，对于停止词’<|im_end|>‘（共10个字符），生成器会在生成文本过程中不断检查最后生成的10个字符是否与停止词相同，即判断cur_text[-10:]是否等于’<|im_end|>'。如果满足条件，则生成过程会立即停止。这样一来，我们便能根据设定的停止词来控制生成文本的范围和内容。

# 细节三：生成下一个词的方式
在自然语言处理中，语言模型的生成通常有以下几种主要方法：

贪心法（Greedy Search）：在每一步，模型都选择概率最高的单词作为输出，并将其作为下一步的输入。优点是计算复杂度低，但可能会过早地陷入局部最优解，生成的文本质量和多样性可能较差。
集束搜索（Beam Search）：在每一步，模型不仅保存概率最高的单词，还保存概率次高的若干个单词。之后会基于这些单词生成多个潜在的句子，并选择总体概率最高的句子。在许多任务中表现较好，但可能生成不完整或语义不明确的句子。
随机采样（Random Sampling）：在每一步，根据模型输出的概率分布随机选择单词。这能保证生成文本的多样性，但可能会生成语法错误或不连贯的文本。
顶部采样（Top-k / Top-p Sampling）：只考虑模型输出概率最高的顶部k个单词，或者选择概率累积到0.9（或者其他阈值p）为止的单词进行随机采样。这在一定程度上平衡了文本的质量和多样性。
model.generate函数的默认设置是使用贪心搜索（Greedy Search），即在每一步选择概率最高的词作为下一个词。

然而，这个函数还包含了许多参数可以修改生成方法，包括：

集束搜索（Beam Search）：通过设置num_beams参数大于1可以启用集束搜索，num_beams的值表示集束的宽度。
随机采样（Random Sampling）：通过设置do_sample=True和temperature参数（调整分布的锐度）可以启用随机采样。
顶部采样（Top-k /Top-p Sampling）：通过设置top_k（选择有最高概率的前k个词）和top_p参数（选择累计概率大于某个阈值p的词）可以启用顶部采样。
配置代码如下：
```python
# 配置generate
temperature=0
max_new_tokens=2048
top_k=40
top_p=0.75
gen_config = GenerationConfig(
    max_new_tokens=max_new_tokens,
    do_sample=temperature > 0,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
)
```
# 细节四：将结果解码
模型生成的结果中，包含了prompt（提示）部分的内容。但在分析模型的输出结果时，我们关注的是模型对于提示的回应，即模型生成的答案部分。因此，需要从完整的输出结果中去除prompt部分，仅保留模型输出的回答内容。

代码：
```python
# 创建一个CSV文件来存储结果
count = 0
with open("results.csv", "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["input", "model_output", "label"])

    for example in data['train']["conversation"]:
        for turn in example:
            count += 1
            if count %10 == 0:
                print(count)
            input_text = turn["input"]
            label = turn["output"]
            # print('===input_text===',input_text)
            # print('===label===',label)
            ids = tokenizer.encode(input_text, return_tensors='pt')
            output = generate_output = model.generate(
                        inputs=ids.cuda(),
                        generation_config=gen_config,
                        stopping_criteria=stop_criteria
                        )
            result = tokenizer.decode(output[0][len(ids[0]):])
            csv_writer.writerow([input_text, result, label])
```
