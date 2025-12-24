# 一、实验模型对于翻转行列之后的表格问题是否还能正确回答

第一部分是针对hitabQA数据集中前50个样例中原本模型预测正确的样例进行的实验。随机种子设置为333。

前50个样例都是一些小表格和简单问题。其中hitab_test_changed.json是从hitabQA数据集前50条中提取的预测正确的原始数据。hitab_test_changed_1.json是根据hitab_test_changed.json进行了简单修改的数据集（使用豆包大模型进行的修改，人工抽查发现修改的没问题），这个数据集仅对hitab_test_changed.json的原数据进行一次行或列的对换。hitab_test_changed_2.json是在hitab_test_changed_1.json的基础上又进行了修改，这次修改力度较大，对换的行列数较多，同时由于豆包大模型本身对于表格理解的不足，出现了改变表格本质的修改，即检索同样的表头，单元格的值出现了不同（但是与问题有关的单元格没有发生变化）。后缀为answer.json的是对应的推理结果。所有预测结果均正确，即原来能够预测正确的数据，进行行列变化甚至是改变某些无关单元格的值均不影响结果。实验发现该模型在小数据上对于数据的变换造成的扰动能够很好适应，不仅适应行列的对换，还能适应某些单元格值的变化。

观察数据集发现，其QA大多都是有固定答案的问题，类似提取式回答，部分涉及数值计算但是答案也是固定的。通过本实验发现只要是模型预测正确的表格QA数据，对其进行行列调整和少量数据修改不影响结果，甚至使用严格判等的方法判断是否预测正确也是一样的。

这里对他原来的推理代码进行了一定修改，比如删除了使用flash_attn的代码（服务器用不了），out = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)，他原来两个False，现在改成True。两个False的话用严格判等正确率为0，不懂什么情况。

inference_hitab_tabfact_fetaqa.py是推理代码，mytest.sh是运行该代码的脚本。

第二部分是针对tabfact数据集进行的实验。同样随机种子设置333。对完整的原始样本进行复现的准确率是0.8241646451209015。（10532/12779）

使用原数据集前30个样例中预测正确的数据进行实验。tabfact_1.json是对原始数据进行列对换得到的数据，没有进行行对换。tabfact_2.json是在tabfact_1.json基础上进行行对换，而不进行列对换。（防止大模型同时做行列对换出错）

实验发现，使用初始数据预测正确的23个实例。也就是tabfact_0.json，为直接从原始数据提取出来的json文件。

（1）tabfact_1.json。进行随机列对换仅2个实例预测错。 **其中有一个验证了一下发现是原始数据标注错了，而且模型预测到了那个错误的答案。** 

    "idx": 12,
    
    "instruction": "This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or refuted by the given table.",
    
    "input_seg": "[TLE] The table caption is about united states national rugby union team. [TAB] | conv | player | span | start | tries | pens | drop [SEP] | 0 | vaea anitoni | 1992 - 2000 | 44 | 26 | 0 | 0 | [SEP] | 0 | paul emerick | 2003 - 2012 | 49 | 17 | 0 | 0 | [SEP] | 0 | todd clever | 2003 - | 51 | 11 | 0 | 0 | [SEP] | 0 | philip eloff | 2000 - 2007 | 34 | 10 | 0 | 0 | [SEP] | 0 | takudzwa ngwenya | 2007 - | 27 | 10 | 0 | 0 | [SEP] | 14 | chris wyles | 2007 - | 35 | 10 | 22 | 1 | [SEP] | 0 | david fee | 2002 - 2005 | 28 | 9 | 0 | 0 | [SEP] | 90 | mike hercus | 2002 - 2009 | 45 | 9 | 76 | 4 | [SEP] | 0 | riaan van zyl | 2003 - 2004 | 12 | 9 | 0 | 0 |",
    
    "question": "The statement is:  <riann van zyl have the shortest time span on the united state national rugby union team and tie with 3 others for the smallest number of tries>. Is it entailed or refuted by the table above?",
    
    "output": "entailed",
    
    "predict": "entailed",

豆包和智谱清言（使用了深度思考的，可信度较高）认为该样例是错误。

（2）tabfact_2.json。再进行行对换仅1个实例预测错。

    "idx": 12,
    
    "instruction": "This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or refuted by the given table.",
    
    "input_seg": "[TLE] The table caption is about united states national rugby union team. [TAB] | conv | player | span | start | tries | pens | drop [SEP] | 0 | riaan van zyl | 2003 - 2004 | 12 | 9 | 0 | 0 | [SEP] | 0 | paul emerick | 2003 - 2012 | 49 | 17 | 0 | 0 | [SEP] | 0 | todd clever | 2003 - | 51 | 11 | 0 | 0 | [SEP] | 0 | philip eloff | 2000 - 2007 | 34 | 10 | 0 | 0 | [SEP] | 0 | takudzwa ngwenya | 2007 - | 27 | 10 | 0 | 0 | [SEP] | 14 | chris wyles | 2007 - | 35 | 10 | 22 | 1 | [SEP] | 0 | david fee | 2002 - 2005 | 28 | 9 | 0 | 0 | [SEP] | 90 | mike hercus | 2002 - 2009 | 45 | 9 | 76 | 4 | [SEP] | 0 | vaea anitoni | 1992 - 2000 | 44 | 26 | 0 | 0 |",
    
    "question": "The statement is:  <riann van zyl have the shortest time span on the united state national rugby union team and tie with 3 others for the smallest number of tries>. Is it entailed or refuted by the table above?",
    
    "output": "entailed",
    
    "predict": "refuted",

这个给大模型再次检查发现就是refuted。

这个数据对应的原始数据如下（在原始的tabfact_test.json内，对应tabfact_0.json第13条，从1开始数）：

      "input": "[TLE] The table caption is about united states national rugby union team. [TAB] | player | span | start | tries | conv | pens | drop [SEP] | vaea anitoni | 1992 - 2000 | 44 | 26 | 0 | 0 | 0 | [SEP] | paul emerick | 2003 - 2012 | 49 | 17 | 0 | 0 | 0 | [SEP] | todd clever | 2003 - | 51 | 11 | 0 | 0 | 0 | [SEP] | philip eloff | 2000 - 2007 | 34 | 10 | 0 | 0 | 0 | [SEP] | takudzwa ngwenya | 2007 - | 27 | 10 | 0 | 0 | 0 | [SEP] | chris wyles | 2007 - | 35 | 10 | 14 | 22 | 1 | [SEP] | david fee | 2002 - 2005 | 28 | 9 | 0 | 0 | 0 | [SEP] | mike hercus | 2002 - 2009 | 45 | 9 | 90 | 76 | 4 | [SEP] | riaan van zyl | 2003 - 2004 | 12 | 9 | 0 | 0 | 0 |",

      "question": "The statement is:  <riann van zyl have the shortest time span on the united state national rugby union team and tie with 3 others for the smallest number of tries>. Is it entailed or refuted by the table above?",
  
      "output": "entailed",
    
应该是refuted。

# 二、测试模型是否“作弊”

（1）测试他训练好的tablellama模型。

主要对数据集进行如下调整：人工找到目标单元格，对数据进行替换，问题不变；人工找到目标单元格所在的行或列，对整行整列数据进行替换，问题不变；同步修改表头和问题，单元格不变。修改后的数据集为hitab_test_changed_new.json，包含五条数据。

**没有做删除一行或一列的数据，因为这会显著改变表格结构，而仅修改某个表头的值或者问题能够最大化地进行混淆，测试模型是否用了测试集“作弊”**

具体修改如下：在hitab_test_changed.json，也就是模型初始预测正确的那些数据上修改。第一个样例把目标单元格52.1改成67.1；第二个样例把55.0改成55.01（极度相似）；第三个样例直接把两个北部的列，表头改成南部，正常来说模型应该输出“找不到”，看看效果如何；第四个样例把问题改成问最低比例的，表格完全不变，问题也基本不变。（只把higher改成lower）；第五个样例直接把非洲换成美洲（行表头），正常来说模型应该找不到。

具体效果在20251224实验预测效果.json。可以看到，效果很差。（404就是表格里面没有答案）

现在依次分析：

1和2两个样例，仅仅对目标单元格的值进行了替换，没有修改其他单元格和问题。之前可以预测对，现在还是能预测对修改之后的值。

3这个样例，模型输出了一个60.6，但是实际上所问的问题在表格里面根本找不到。虽然如此，输出的60.6也不是修改前的正确答案。

4这个样例，我把问最高改成问最低，模型直接输出了一个空字符串。正确答案应该是其他地区。

特别是这个第五条，在把行表头换成美洲之后，正常应该是找不到，但是模型就开始乱回答了，但是也没有输出修改之前的结果。

**可以得出结论，由于进行了大量的表格的微调，更重要的是，同样的一个表格上平均做了4个不同的问题进行训练，根据实验结果，模型对于单个表格的机械记忆情况应该是不存在的。**

值得注意的是，模型乱回答的情况依然严重，特别是对于在表格中找不到答案的情况，模型处理起来效果很差。后面造数据集可以适当设置部分在表格里面找不到答案的数据对模型进行增强。
