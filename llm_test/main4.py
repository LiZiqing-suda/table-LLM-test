from openai import OpenAI
import json
import time
with open(r"D:\论文\eval_data_initial\in_domain_test\hitab_test.json","r",encoding="utf-8") as f:
    d = json.load(f)


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="?????"
)

right = 0
w = 0
try:
    for i in range(len(d)):
        try:
            content = "instruction:"+d[i]["instruction"]+" "+"input_seg:"+d[i]["input_seg"]+"question:"+d[i]["question"]
            content1="请根据这些信息，自行解读表格结构，用英文或数字回答问题，其中input_seg是表格。仅需要输出最终回答的答案，不要保留推断内容、依据等中间过程。答案应该是确定性的，且尽量简短，仅包含必要信息。请不要在数值答案前面加-，除非真的是负数"
            content=content1+content
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.1-terminus",
                messages=[{"role": "user", "content": content}],
                temperature=0.01,
                top_p=0.9,
                max_tokens=1024,
                stream=False
            )
            time.sleep(0.01)
            res = completion.choices[0].message.content
            #print(res)
            c = "根据以下信息，判断下面的回答是否正确（answer部分）。这里提供了标准答案，你的任务是根据标准答案（output部分），判断answer部分回答是否正确（可以表达不同但是意思要一样，如果是好几个并列的名词，顺序无所谓），如果回答是数字，只要数值一样都算对（浮点数如果相差结果在1e-6以内算对，否则是错的），不管输出的浮点数小数还是科学计数法，有的数字用了逗号分，也认为一样，比如1036和1,036认为一样。你认为正确输出1，错误输出0。仅可以输出1或者0，且不要有多余输出，禁止输出任何中间推导过程。"
            c += "\nanswer:" + res + " "
            c += "\noutput:" + d[i]["output"]
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.1-terminus",
                messages=[{"role": "user", "content": c}],
                temperature=0.0001,
                top_p=0.9,
                max_tokens=1024,
                stream=False
            )
            res1 = completion.choices[0].message.content
            if res1 == "1":
                right += 1
            elif res1 == "0":
                w += 1
            print(res1)
            print(int(res1),end=" "+d[i]["output"])
            print()
            time.sleep(0.01)
        except Exception:
            print("w")
            pass
except:
    print("right:", end=str(right))
    print()
    print("wrong:", end=str(w))
    print()
    print(right / (w + right))
finally:
    print("right:", end=str(right))
    print()
    print("wrong:", end=str(w))
    print()

    print(right / (w + right))

