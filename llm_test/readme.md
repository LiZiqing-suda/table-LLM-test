测试市面上的大模型理解表格的能力，开源闭源都测一下。数据集为fetaqa数据集，是复杂问题数据集，每个问题没有固定答案（语义是固定的但是回答形式可以多样）。由于人工查输出答案对不对太慢了，直接让大模型自己检查。这里所有的测试都关闭了大模型的深度思考，因为后面对大模型进行微调等任务均用不了深度思考。

（1）使用nvidia的deepseekV3.1的api进行了实验。先实验自己输出的答案自己会不会反驳掉。实验代码为main.py。实验思路就是把表格和指令输入到该大模型，得到一个输出，再将表格、指令和输出再次输入到大模型让他判断上次的输出对不对。（两次均不提供标准答案output，让模型自己判断）

结果如下：

    相同1848
    
    不同144
    
    相同的比例0.927710843373494

也就是自己生成的答案再给同样的模型判断答案是否正确，92.77%的情况第二次模型认为第一次的输出是正确的。也就是两次调用大模型（仅修改了T，第一次0.2第二次0.01，但是其实都差不多，0.2已经很小了），同样的数据有7%多的情况两次预测结果本质不同（至少有一个是错的）。

（2）再进行交叉的实验。a模型生成输出让b模型判断，同时b模型生成输出让a模型判断。这次换硅基流动的api，测试deepseek-ai/DeepSeek-V3.1-Terminus（685B MoE 模型）和Qwen/Qwen2.5-72B-Instruct。总共2003个样本，但是他这个api调用不稳定，加了time.sleep(0.3)还是出现大量调用失败的情况。

deepseek-ai/DeepSeek-V3.1-Terminus写，Qwen/Qwen2.5-72B-Instruct判断（main1.py）：

    right:285
    wrong:5
    0.9827586206896551

Qwen/Qwen2.5-72B-Instruct写，deepseek-ai/DeepSeek-V3.1-Terminus判断（main2.py）：

    right:1843
    wrong:160
    0.9201198202695956

（3）这次将数据集的标准output输入进去。具体是第一次调用不输入output，而是输入表格、指令和问题，让模型自己预测一个结果，第二次把表格、指令、问题、第一次的输出和标准output全部输入，让模型判断第一次的输出是否正确，其实就是代替人工检查大模型的输出对不对。由于有标准output数据，加上大模型自己的能力，这个第二次的判断我们可以认为是一定准确的。用deepseek-ai/DeepSeek-V3.1-Terminus。

    right:734
    wrong:139
    0.8407789232531501

（4）补充测试一下简单的数据集。hitab_test数据集，看看tablellama号称的SOTA准确率和大模型相比如何。该数据集有较多数值计算的任务，这是大模型的短板，所以效果比fataqa数据集低很多。

    right:1057
    wrong:526
    0.6677195198989261

大概在这个附近，让大模型判断标准output和预测答案也有少量出错的情况。

以下实例均为nvidia网站的网页版deepseek-v3.1-terminus，关闭深度思考模式。

实例一：

        请根据这些信息，自行解读表格结构，用英文或数字回答问题，其中input_seg是表格。仅需要输出最终回答的答案，不要保留推断内容、依据等中间过程。答案应该是确定性的，且尽量简短，仅包含必要信息。请不要在数值答案前面加-，除非真的是负数。
        "instruction": "This is a hierarchical table question answering task. The goal for this task is to answer the given question based on the given table. The table might be hierarchical.",
        "input": " [TLE] The table caption is percentage of canadian-born black immigrants aged 25 to 59 with a postsecondary diploma, by sex and region or country of ancestry, 2016. [TAB] | region or country of ancestry | women | women | men | men | [SEP] | region or country of ancestry | non-university or university postsecondary diploma | university degree only | non-university or university postsecondary diploma | university degree only | [SEP] | region of ancestry | percent | percent | percent | percent | [SEP] | caribbean and latin america | 78.1 | 34.8 | 59.5 | 18.4 | [SEP] | africa | 79.6 | 50.8 | 63.6 | 35.3 | [SEP] | other regions | 59.8 | 19.1 | 46.8 | 14.1 | [SEP] | country of ancestry |  |  |  |  | [SEP] | jamaica | 75.5 | 31.3 | 54.6 | 15.8 | [SEP] | haiti | 84.5 | 37.1 | 65.8 | 18.3 | [SEP] | trinidad and tobago | 76.1 | 36.5 | 61.4 | 20.4 | [SEP] | barbados | 79.4 | 39.2 | 64.4 | 22.8 | [SEP] | guyana | 74.8 | 32.5 | 57.6 | 20.6 | [SEP] | saint vincent and the grenadines | 76.1 | 35.0 | 58.3 | 17.5 | [SEP] | grenada | 78.0 | 34.3 | 62.0 | 21.4 | [SEP] | ghana | 81.7 | 49.8 | 65.7 | 31.4 | [SEP] | nigeria | 86.4 | 63.3 | 75.3 | 51.5 | [SEP] | united states | 63.9 | 23.2 | 49.2 | 17.7 | [SEP] | united kingdom | 71.9 | 32.0 | 53.9 | 22.1 | [SEP] | canada | 56.5 | 16.0 | 44.8 | 11.8 |",
        "question": "how many percent of university graduates among second-generation black women who originated from jamaica was higher than that of men in 2016?",
    
T=0.01,top_p=0.9。回答31.3，答案是15.5。尝试了多个seed都是这样。

实例二：

        请根据这些信息，自行解读表格结构，用英文或数字回答问题，其中input_seg是表格。仅需要输出最终回答的答案，不要保留推断内容、依据等中间过程。答案应该是确定性的，且尽量简短，仅包含必要信息。请不要在数值答案前面加-，除非真的是负数
        "instruction": "This is a hierarchical table question answering task. The goal for this task is to answer the given question based on the given table. The table might be hierarchical.",
        "input": " [TLE] The table caption is percentage of canadian-born black immigrants aged 25 to 59 with a postsecondary diploma, by sex and region or country of ancestry, 2016. [TAB] | region or country of ancestry | women | women | men | men | [SEP] | region or country of ancestry | non-university or university postsecondary diploma | university degree only | non-university or university postsecondary diploma | university degree only | [SEP] | region of ancestry | percent | percent | percent | percent | [SEP] | caribbean and latin america | 78.1 | 34.8 | 59.5 | 18.4 | [SEP] | africa | 79.6 | 50.8 | 63.6 | 35.3 | [SEP] | other regions | 59.8 | 19.1 | 46.8 | 14.1 | [SEP] | country of ancestry | | | | | [SEP] | jamaica | 75.5 | 31.3 | 54.6 | 15.8 | [SEP] | haiti | 84.5 | 37.1 | 65.8 | 18.3 | [SEP] | trinidad and tobago | 76.1 | 36.5 | 61.4 | 20.4 | [SEP] | barbados | 79.4 | 39.2 | 64.4 | 22.8 | [SEP] | guyana | 74.8 | 32.5 | 57.6 | 20.6 | [SEP] | saint vincent and the grenadines | 76.1 | 35.0 | 58.3 | 17.5 | [SEP] | grenada | 78.0 | 34.3 | 62.0 | 21.4 | [SEP] | ghana | 81.7 | 49.8 | 65.7 | 31.4 | [SEP] | nigeria | 86.4 | 63.3 | 75.3 | 51.5 | [SEP] | united states | 63.9 | 23.2 | 49.2 | 17.7 | [SEP] | united kingdom | 71.9 | 32.0 | 53.9 | 22.1 | [SEP] | canada | 56.5 | 16.0 | 44.8 | 11.8 |",        
        "question": "what is the difference between black women with university degree only who originated from haitian and that of men?",

T=0.01,top_p=0.9。回答2.8，答案是18.8。
