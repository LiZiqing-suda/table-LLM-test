# **自己做的表格理解数据集。**

爬了ICLR2024和2025所有挂在arxiv以及有html版本的论文。把他们的html文件爬下来并且用python代码对三线表进行了提取。数据集制作的中间结果存在tmpdataset目录下。

**数据集格式：**

    （1）每个以v1、v2这种版本号结尾的zip文件是一整个数据集。每加工处理一次就提交一个新的版本上来。
    
    （2）每个数据集里面有若干个目录，以arxiv号命名，每个目录对应一篇论文的提取结果。
    
    （3）每个以arxiv号命名的目录下有一个captions.txt文件，记录表格的标题名称。然后有多个csv文件，是表格的内容。表格的编号是按html里面的先后顺序进行
    
    的编号，正常情况下也就是pdf文件当中的先后顺序。还有多个txt文件，除了captions.txt以外均为论文中引用到这些表格的语句，一行一句。如Table_1.txt为所
    
    有引用到Table 1的句子。还有个多表引用句子的txt，为Multi-table.txt。每个句子只会在一个txt文件中出现。

    （4）csv的命名方式是Table_num.csv（如Table_2.csv、Table_14.csv），

v1：初步进行了提取，还没有做验证和整理，仅提取了表格内容及其对应的标题。表格内容可能有因为html渲染语法的原因出现少数提取错误的情况，后续需人工筛选验证。暂未对文章引用这些表格的语句进行提取。

v2：对html内所有引用到这些表格的语句进行了提取。具体提取的内容是：从提到该表格的句子开始，到这一段结束，均归类到该表格对应语句的txt内。如果提到多表格，则优先放在Multi-table.txt。每个句子只会在一个txt文件中出现。

如It is difficult for non-expert users to assess the accuracy of the generated code, we automatically utilize the Example information to verify the accuracy of the CoNN model - checking whether the output result of the input sequence is exactly consistent with the Example. The results shown in **Table 4** demonstrate that generally 2 Examples are sufficient to select an accurate CoNN model, which means it is very easy for users to use and demonstrate. However, considering the varying difficulty of different tasks, we still suggest non-expert users provide more Examples to ensure the accuracy of the generated CoNN.

这段话提取的内容是The results shown in **Table 4** demonstrate that generally 2 Examples are sufficient to select an accurate CoNN model, which means it is very easy for users to use and demonstrate. However, considering the varying difficulty of different tasks, we still suggest non-expert users provide more Examples to ensure the accuracy of the generated CoNN.

存放在Table_4.txt内。
