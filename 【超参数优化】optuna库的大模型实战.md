- [optuna是什么](#optuna是什么)
- [超参数优化是什么](#超参数优化是什么)
- [optuna实战](#optuna实战)

# optuna是什么
Optuna是一个开源的超参数优化框架，Optuna与框架无关，可以在任何机器学习或深度学习框架中使用它。

# 超参数优化是什么
超参数优化（Hyperparameter Optimization，简称HPO）是指自动调整机器学习模型中的超参数以提升模型性能的过程。在深度学习和机器学习中，模型的训练涉及两类参数：模型参数和超参数。

模型参数：模型在训练过程中自动学习的参数，例如神经网络中的权重和偏置。

超参数：模型训练前需要预先设置的参数。它们不是通过训练数据直接学习来确定的。例如，神经网络的学习率、层数、每层的节点数，或者决策树的最大深度、最小叶节点样本数等。

为什么要进行超参数优化？
超参数的设定对模型的学习能力和性能有极大影响。不正确地设定超参数可能导致模型学习效率低下或者无法学习到有用的模式；例如，过高的学习率可能导致模型无法收敛，而过低的学习率可能导致学习过程过慢或陷入局部最优解。因此，找到一组合适的超参数对于训练高效、高性能的模型至关重要。

## 超参数优化通常涉及的步骤：

定义超参数空间，即每个超参数可以取值的范围或者可选的值列表。
选择一个性能度量指标，如交叉验证中的准确度或验证集上的损失函数。
运行超参数搜索算法，通过迭代地尝试不同的超参数配置来评估模型性能。
选择最佳的超参数配置，这是在给定度量指标上表现最好的配置。

## 常用的超参数优化方法：

网格搜索（Grid Search）
随机搜索（Random Search）
贝叶斯优化（Bayesian Optimization）
遗传算法（Genetic Algorithms）
梯度下降（Gradient-based Optimization）
超带算法（Hyperband）
Population Based Training (PBT)
超参数优化方法可以大大减轻手动调试超参数的负担，提高模型性能，并提高模型搜索的效率。由于超参数优化通常涉及多次训练和评估模型，该过程可能非常耗时，因此，高效的超参数优化算法对于节省时间和计算资源至关重要。

# optuna实战
## 参考资料
本文分享optuna在大模型领域的实战。用到的代码来自于：

https://github.com/zyds/transformers-code/tree/master

这位up主的大模型领域的教学视频通俗易懂，推荐新人观看：

你可是处女座啊的个人空间-你可是处女座啊个人主页-哔哩哔哩视频

其中关于optuna的视频的链接是：

【【手把手带你实战HuggingFace Transformers-番外技能篇】基于Optuna的transformers模型自动调参】 【手把手带你实战HuggingFace Transformers-番外技能篇】基于Optuna的transformers模型自动调参_哔哩哔哩_bilibili

## 全部代码
本代码使用transformers库来微调huggingface里的"hfl/rbt3"模型（一种bert模型）。代码来源：https://github.com/zyds/transformers-code/blob/master/Others/01-hyp_tune/hyp_tune_optuna.ipynb
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

dataset = load_dataset("csv", data_files="ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

def process_function(examples):
   tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
   tokenized_examples["labels"] = examples["label"]
   return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)

def model_init():
   model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
   return model

import evaluate
acc_metric = evaluate.load("accuracy")
f1_metirc = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=500,               # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

from transformers import DataCollatorWithPadding
trainer = Trainer(model_init=model_init, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()

def default_hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
        "optim": trial.suggest_categorical("optim", ["sgd", "adamw_hf"]),
    }

trainer.hyperparameter_search(hp_space=default_hp_space_optuna, compute_objective=lambda x: x["eval_f1"], direction="maximize", n_trials=10)
```

## 核心代码解读
1.model_init()
```python
def model_init():
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    return model

trainer = Trainer(model_init=model_init, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)
```
注意，这里传进Trainer里的是model_init，而非model，在Hugging Face的transformers库中，model_init通常指的是一个用户提供的函数，用于初始化（或重新初始化）一个模型。这在超参数搜索或者当想要多次用随机初始化的模型训练数据（例如，为了评估初始化的不确定性）时特别有用。由于一些训练和超参数搜索过程可能需要多次重新开始训练，model_init使得你可以在每次搜索试验或训练运行前重新生成一个干净的模型。

超参数搜索一般会跑多次模型，每次需要从头开始跑，而不是在上一次实验的基础上开始跑。这就是model_init的用途。



2.trainer.train()

注意，在代码的最后一行，是开始超参数调试（即训练若干个不同的模型，根据指标效果，找到最合适的超参数）。在正式的超参数调试之前，模型先跑了一遍trainer.train()。为什么要先跑一遍模型？

如果没有先跑一遍模型，在跑超参数优化的环节会报如下warning：

Some weights of BertForSequenceClassification were not initialized from the model checkpoint at hfl/rbt3 and are newly initialized: ['classifier.bias', 'classifier.weight']

这个warning传递了两个信息：

（1）某些权重未被初始化：在从模型检查点hfl/rbt3加载模型时，有些权重没有从预训练模型中初始化而是被新创建了。具体来说，是属于用于序列分类任务的classifier层的权重（classifier.bias和classifier.weight）。这种情况通常发生在预训练模型没有附属特定任务的头结构（在本例中是分类头），或者被加载的检查点与模型架构不完全匹配时。这意味着在该分类层中的权重是随机初始化的，需要通过下游任务进行训练才能得到有意义的值。

（2）建议训练模型：建议应该在一个下游任务上训练这个模型，以便能够用它进行预测和推理。换句话说，虽然模型的主体部分（基于BERT）已经被预训练了，但是新增加的分类层是未训练的，因此，在使用这个序列分类模型之前，需要在具有标签的数据上进行训练，即进行"fine-tuning"(微调)。

换句话说，完整的模型包括位于最后的classifier层，但是加载的huggingface的预训练的模型并未包括这一层。因此，在我们加载 Hugging Face 的预训练模型时，该分类层的权重是从头开始随机初始化的。接下来，在进行超参数优化时，每次的实验都会以这个分类层拥有随机初始权重的模型为基础进行微调。但如果我们在进入超参数优化之前就执行了trainer.train()方法，那么这个分类层就已经经历了初步的微调，它的权重也就不再是随机设定的了。

根据我的个人实践经验，我会推荐在进入超参数搜索阶段之前，先行运行一次trainer.train()以确保模型不是以一个具有随机权重的状态开始搜索过程。

3.trainer.hyperparameter_search
```python
trainer.hyperparameter_search(
hp_space=default_hp_space_optuna, 
compute_objective=lambda x: x["eval_f1"], 
direction="maximize", 
n_trials=10)
```
不同于trainer.train()，trainer.hyperparameter_search是用来执行超参数搜索的函数。

我们点进去看看源码：
```python
 def hyperparameter_search(
 self,
 hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
 compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
 n_trials: int = 20,
 direction: Union[str, List[str]] = "minimize",
 backend: Optional[Union["str", HPSearchBackend]] = None,
 hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> Union[BestRun, List[BestRun]]:
```
第一个参数hp_space是一个定义超参数搜索空间的函数，可选。如果没有指定，则首先默认在[OptunaBackend, RayTuneBackend, SigOptBackend, WandbBackend]这四个backend里按顺序选择一个backend，然后根据backend来选择默认的模板。

这里的backend指的是超参数搜索的后端，比如optuna就是一个后端。如果没有显示指定，那么optuna就是默认的后端。

Optuna后端的源码预设了一套标准的超参数搜索空间（hp_space），如以下JSON对象所示：
```python
{
 "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
 "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
 "seed": trial.suggest_int("seed", 1, 40),
 "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
}
```
可以看到这里指定了四种需要通过做实验来确定的超参数——learning_rate、num_train_epochs、seed、per_device_train_batch_size。

例如，trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)指出learning_rate将在1e-6到1e-4的对数尺度区间内进行浮点类型的随机取值。而trial.suggest_int的使用方式与此类似，只不过是针对整数类型的随机取值。

对于希望从给定的离散选项集中挑选最优项的情况，我们使用trial.suggest_categorical方法。例如，在这里，为了决定最合适的批处理大小（batch size），我们将从[4, 8, 16, 32, 64]这五个备选值中进行选择和实验，以寻找可能的最优解。

你也可以新增其他想要实验的超参数，一般来说，在TrainingArguments里面可以定义的超参数都可以作为想要实验的超参数，送到default_hp_space_optuna里。

第二个参数是compute_objective，可选。其作用是利用evaluate方法反馈的度量标准来计算目标函数值，以便进行最小化或最大化优化。若在调用时没有特定的评估指标提供，该函数便会默认使用评估损失作为返回结果；若有提供，则返回这些指标之和。在本例中，我们采用验证集的 F1 值作为衡量标准。

第三个参数 direction 指明了我们希望目标函数（compute_objective）是追求更大的值（"maximize"）还是更小的值（"minimize"）。考虑到使用了 F1 值作为评价标准，我们自然是追求其最大化。因此，这里的设定应为 "maximize"。

第四个参数 n_trials 指定了试验的次数，其默认值为 20 次。该参数决定了在超参数搜索过程中将尝试并评估多少种不同的参数组合。通过设置合适的试验次数，可以在探索的广度与深度间寻找平衡，旨在找到最优的超参数集。

## 代码运行结果
每次实验运行结束后，都会显示当前实验的信息，以及历史最好的实验编号：

[I 2024-01-12 03:10:22,378] Trial 0 finished with value: 0.8975957257346394 and parameters: {'learning_rate': 3.944651363100783e-06, 'num_train_epochs': 3, 'seed': 38, 'per_device_train_batch_size': 32, 'optim': 'adamw_hf'}.
Best is trial 0 with value: 0.8975957257346394.
在所有实验结束后，输出：

BestRun(run_id='6', objective=0.9170383586083853, hyperparameters={'learning_rate': 2.6823388434045424e-05, 'num_train_epochs': 3, 'seed': 1, 'per_device_train_batch_size': 32, 'optim': 'adamw_hf'}, run_summary=None)
代表最优的实验编号是6.

在checkpoints文件夹中，有每次实验的保存checkpoints.
