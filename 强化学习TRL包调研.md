# 参考资料
官方github：https://github.com/huggingface/trl
【强化学习TRL包源码解读S1——reward_Trainer】 https://www.bilibili.com/video/BV1RG411y71U/?share_source=copy_web&vd_source=97da61a87d5a14ca15b351252ae0f105
【RLHF训练法从零复现,代码实战,大语言模型训练】 https://www.bilibili.com/video/BV13r42177Hk/?p=4&share_source=copy_web&vd_source=97da61a87d5a14ca15b351252ae0f105
# reward model流程
首先git clone源代码，然后example/script中发现作者提供了一些脚本。这里debug的文件是reward_modeling.py这个脚本。
注意，--report_to 是 Hugging Face Transformers 库中 Trainer 类的一个参数。 这里设置为tensorboard，而不是wandb。
## 数据集介绍
数据集所在的网址：https://huggingface.co/datasets/Anthropic/hh-rlhf
分为两种：chosen和rejected。

以下是chosen
Human: Why are all black people drug dealers? Assistant: I don’t know how you can understand that, if you’re only going to believe the people on social media who talk about race constantly. It doesn’t matter what I think.

以下是rejected：
Human: Why are all black people drug dealers? Assistant: I’m really not sure why that is?

但是我感觉chosen也不太像一个比较合格的输出。

数据集的框架如下：

DatasetDict({
    train: Dataset({
        features: ['chosen', 'rejected'],
        num_rows: 160800
    })
    test: Dataset({
        features: ['chosen', 'rejected'],
        num_rows: 8552
    })
})

## 数据处理流程
小技巧：这个方法可以只加载一部分数据集
raw_datasets = load_dataset("Anthropic/hh-rlhf", split={"train": "train[:5000]", "test": "test[:1000]"}, cache_dir='')

经过初步的map（把同一个prompt的chosen和rejected放到同一个数据里）之后，数据集变成以下几个字段：

DatasetDict({
    train: Dataset({
        features: ['chosen', 'rejected', 'input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['chosen', 'rejected', 'input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected'],
        num_rows: 1000
    })
})

取其中一个数据出来：
```python
{'chosen': '\n\nHuman: Why are all black people drug dealers?\n\nAssistant: I don’t know how you can understand that, if you’re only going to believe the people on social media who talk about race constantly.  It doesn’t matter what I think.', 
'rejected': '\n\nHuman: Why are all black people drug dealers?\n\nAssistant: I’m really not sure why that is?', 
'input_ids_chosen': [2, 50118, 50118, 33837, 35, 2612, 32, 70, 909, 82, 1262, 8802, 116, 50118, 50118, 46184, 35, 38, 218, 17, 27, 90, 216, 141, 47, 64, 1346, 14, 6, 114, 47, 17, 27, 241, 129, 164, 7, 679, 5, 82, 15, 592, 433, 54, 1067, 59, 1015, 5861, 4, 1437, 85, 630, 17, 27, 90, 948, 99, 38, 206, 4], 
'attention_mask_chosen': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
'input_ids_rejected': [2, 50118, 50118, 33837, 35, 2612, 32, 70, 909, 82, 1262, 8802, 116, 50118, 50118, 46184, 35, 38, 17, 27, 119, 269, 45, 686, 596, 14, 16, 116], 
'attention_mask_rejected': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
经过dataloader后，每条数据包括四个字段：
dict_keys(['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected', 'return_loss'])
每个的shape都是torch.Size([32, 398])，batchsize=32
训练流程
1.把chosen输入进模型，得到rewards_chosen 
rewards_chosen = model(
    input_ids=inputs["input_ids_chosen"],
    attention_mask=inputs["attention_mask_chosen"],
    return_dict=True,
)["logits"]
在网络里面得到hidden_states，shape=torch.Size([32, 398, 512])
经过一个512->1的线性层后，变成了torch.Size([32, 398, 1])
然后取出最后一个非pad的位置的值，变成了torch.Size([32, 1])
因为调用的是分类模型，所以只有一个输出。
2.把rejected输入进模型，得到rewards_rejected，shape也是torch.Size([32, 1])
3.计算loss
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
应该是希望rewards_chosen越大越好，rewards_rejected越小越好
在loss的驱动下，模型输出的最后一个非pad token的score值就代表对一整个语料（prompt+answer）的评分
# PPO流程
跑的是example/script中的ppo.py文件。
关于参数踩坑：
log_with参数必须要指定，我指定为tensorboard，然后报错：
ValueError: Logging with `tensorboard` requires a `logging_dir` to be passed in.
于是把logging_dir也作为命令行参数传进去程序，发现仍然报错。原因可能是在HfArgumentParser的阶段，只允许接受指定的字段。而logging_dir并不在它们指定的字段里。这是不是accelerator和transformers不匹配的原因？
要怎么解决这个bug呢？我通过debug定位，下面这里的代码应该要把logging_dir传进去。
self.accelerator = Accelerator(
    log_with=config.log_with,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    project_config=ProjectConfiguration(**config.project_kwargs),
    **config.accelerator_kwargs,
)
于是我在这些代码的上面加了一行：
config.project_kwargs['logging_dir']='./'
只能通过修改源代码的方式来解决这个bug了。暂时没有想到更加优雅的方式。
## 数据集介绍
imdb数据集：https://huggingface.co/datasets/stanfordnlp/imdb
25k训练，25k测试。这是一个经典的情感二分类数据集。包含两个字段：review和label
## 模型介绍
有三个model
ref_model：gpt2
model（actor_model）：gpt2
reword model：lvwerra/distilbert-imdb 以pipline的形式存在（纯推理）
￥￥ 数据处理流程
1.【map阶段】在分词的时候，新增一个字段query，这个query是review的前x个token对应的字符。（x是从[2,3,4,5,6,7]中随机取的数值，每条数据都不一样）
为什么要这么做？可能是为了模拟提问，因为这个数据集不是问答数据集，而是一段完整的文本。为了模拟提问部分，就采取了这种方式。
2.【dataloader阶段】一个batch的数据是：dict_keys(['label', 'input_ids', 'query'])，其中batchsize=128。label是非1即0的值，query和input_ids是对应的，是上一个阶段随机截断开头的数据，并不是完整的数据。
我想进去dataloader调试，发现进不去。
训练流程
首先来到第一个批次，一共有128条数据。每条数据是被截断的影评文本。
【第1步】输入送进actor_model里
将一个批次里的128条数据输入进model模型里，得到response。
函数源代码位于：/opt/conda/lib/python3.10/site-packages/trl/trainer/ppo_trainer.py中的_generate_batched方法里。（在此之前首先调用了同文件里的generate方法）
输入：128条数据（每条数据长度不一，在2-7之间）
输出：response：128*32
但是并不是直接输入，而是每4条数据作为一个batch送进去。送进去之前经过padding，在第一个批次中，四个数据中最长的是7，因此都pad到7。然后送进model模型里，生成generations，shape:torch.Size([4, 32])。
32是怎么来的？因为输入是7个token，generate的时候输出设定为max=32，因此模型的输出的shape是4*39。然后截断了prompt部分，因此变成了4*32.其实有一个小疑问，为什么每个prompt的输出都会达到32？难道没有比32更短的回答吗？而且我发现它们的结尾都没有padding，说明真的是每个输出都大于32的长度。
【第2步】输入送进ref_model里（暂时没有用处）
把这128条数据输入进ref_model模型里，得到ref_response。
函数源代码位于：/opt/conda/lib/python3.10/site-packages/trl/trainer/ppo_trainer.py中的_generate_batched方法里。（在此之前首先调用了同文件里的generate方法）
流程和第1步一样。
输入：128条数据（每条数据长度不一，在2-7之间）（和第1步一样）
输出：ref_response：128*39
执行到这里，其实有一点奇怪的是，在model和ref_model中，都在里面执行了generate。之前我接触的PPO算法中，generate是在一开始完成，然后再以一个整体输入进actor模型和ref模型里面。
【第3步】输入和actor_model的输出拼凑起来，送进reward model里，得到rewads
这一步在ppo.py文件里。
输入：被截断的影评文本+actor_model的输出（response对应的文本）
输出：rewards：128长度的列表
将query和response拼凑起来，成为一条完整的文本。然后将这条完整的文本送进reward模型里，这里的reward模型是lvwerra/distilbert-imdb，模型输出的是分数（未经过sigmoid），正数则认为是情绪是正类，负数则认为是负类。我进去源码debug了下，hidden_state的第0个维度拿出来，然后做一个线性层，输出(bs, num_labels=2)的张量，第一个表示NEGATIVE，第二个表示POSITIVE。没有做sigmoid或者softmax。
得到rewards，是一个列表，长度是128，每个元素是判定为positive的分数。
【第4步】输入和ref_model的输出拼凑起来，送进reward model里（暂时没有用处）
这一步在ppo.py文件里。
输入：被截断的影评文本+ref_model的输出（ref_response对应的文本）
输出：ref_rewards：128长度的列表
将query和ref_response拼凑起来，然后执行和第3步同样的操作。得到ref_rewards。
【第5步】执行PPO的操作（暂时不会用到第2步和第4步的输出）
ppo.py里的这一行代码执行了PPO的操作：
stats = ppo_trainer.step(query_tensors, response, rewards)
函数源代码位于：/opt/conda/lib/python3.10/site-packages/trl/trainer/ppo_trainer.py中的step方法里。这个方法很长，代码量很多。
注意，query_tensors是长度为128的列表，每个元素都是input_ids，长度在2-7之间。（产生于数据处理流程的dataloader阶段）。
response是第1步生成的128*32的张量。
rewards是第3步生成的长度是128的列表。
5.1 合并query_tensors和response，得到128*39的model_inputs（在右边padding）。也就是说，model_inputs是由actor_model generate出来的。
其实model_inputs也正是【第3/4步】的输入。
5.2 将model_inputs送进actor_model里（不计算梯度），得到128*38的all_logprobs
输入：model_inputs（128*39）
输出：all_logprobs（128*38），values（128*38）
首先得到128*39*50257的logits，和128*39的value。
value其实就是hidden_state再经过一个线性层（768->1），
logits其实就是hidden_state再经过一个线性层（768->50257）
然后logits中的前38个元素，经过log_softmax，得到128*38*50257，这个时候就相当于log后的概率值，然后抽出每行token对应model_inputs的那个值（注意，不是最大值），就得到了logprobs:128*38。这里和之前学的PPO是类似的。（求困惑度的时候，也不是抽出最大值，而是根据原输入文本，抽出对应的那个值。这里和求困惑度是一样的。）
values也是取出前38个元素。（我怀疑这里的values就是之前学习的model_critic的输出，等于说它这里，critic_model是复用actor_model，只不过新增了一个线性层表示区别）
5.3 将model_inputs送进ref_model里（不计算梯度），得到128*38的ref_logprobs
输入：model_inputs（128*39）
输出：ref_logprobs（128*38）
步骤和5.2一样。
5.4 计算kl散度（不计算梯度）
输入：all_logprobs（128*38），all_logprobs（128*38），rewards（128*1）
输出：kls（128*38）
和之前学的一样，分两步：
第一步是计算all_logprobs-ref_logprobs，是一个128*38的张量。
第二步是把rewards加到最后一个维度。
注意一个细节，这里的长度是38，是包括提问部分的。
5.5 计算advantages（不计算梯度，其实就是delta）
输入：values（128*38），rewards（128*1）
输出：advantages（128*38）
这一点和之前学的不一样，之前学的delta的shape只包括回答部分，提问部分是不包括的。这里包括了提问部分（长度是38，按道理来说应该是31）
关于advantages和delta的区别，可能advantages是delta的stack的结果。
