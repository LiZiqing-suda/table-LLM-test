#!/bin/bash
# TableLlama 推理脚本（HiTab/TabFact/FetaQA任务）
# 修正：每行末尾反斜杠后无多余空格，参数传递正常

# 执行推理脚本（核心命令，参数严格对齐）
python3 /mnt/home/zlbb/TableLlama/inference_hitab_tabfact_fetaqa.py \
    --base_model /mnt/home/zlbb/tablellama/ \
    --context_size 8192 \
    --max_gen_len 128 \
    --flash_attn False \
    --input_data_file /mnt/home/zlbb/TableLlama/output/hitab_test_changed_answer.json \
    --output_data_file /mnt/home/zlbb/TableLlama/output/111.json

# 错误处理：检查脚本执行结果
if [ $? -eq 0 ]; then
    echo -e "\033[32m✅ 推理执行成功！结果已保存到：/mnt/home/zlbb/TableLlama/output/hitabQA.json\033[0m"
else
    echo -e "\033[31m❌ 推理执行失败！请检查路径/参数是否正确\033[0m"
    exit 1
fi