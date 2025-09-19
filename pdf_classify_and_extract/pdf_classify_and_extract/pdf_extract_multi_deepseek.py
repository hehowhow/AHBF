import json
from openai import OpenAI, BadRequestError
from pypdf import PdfReader
import os
import time
from zai import ZhipuAiClient
import multiprocessing
from functools import partial
import re

# 工艺字典
craft_dict = {
    "PVD": "物理气相沉积 (Physical Vapor Deposition)",
    "FLUX": "熔剂法 (Flux Method)",
    "CVT": "化学气相传输 (Chemical Vapor Transport)",
    "CVD": "化学气相沉积 (Chemical Vapor Deposition)",
    "MOCVD": "金属有机化学气相沉积 (Metal-Organic Chemical Vapor Deposition)",
    "MBE": "分子束外延 (Molecular Beam Epitaxy)",
    "ALD": "原子层沉积 (Atomic Layer Deposition)"
}

craft_prompt_dict = {
    "PVD": '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，依据下列规则对其中的信息进行结构化抽取，并将其归入schema中具体的step。请严格执行每一条指令，不得省略或擅自推断。
      1. 材料提取规则（Material Extraction Rules）：
        - 明确识别论文中所有作为PVD目标材料的实体（即被沉积材料）；
        - 如果论文包含多种被沉积材料的PVD流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的PVD工艺流程**，哪怕部分参数与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段命名为："[material_name]_process"，其中material_name为该材料在论文中出现的**完整英文名称**。

      2. 结构格式要求（Schema Structure Requirements）：
        - 每个step中必须包含字段“step_name”，且该字段始终放在该step的最前面；
        - 除“step_name”以外的字段请严格遵循模板字段结构，没有必要不添加非模板字段；
        - 所有字段值均为英文表达，禁止添加层级标识和注释等内容
        - ！！禁止使用中文！！；
        - 对于step中的子字段，请保持其顺序与论文中描述一致，不可重排

      3. 内容提取规则（Content Extraction Rules）：
        - 所有字段值必须严格基于论文内容提取，不得省略任何能在schema中匹配的字段；
        - 如果论文中提供了实验参数（如温度、气压、功率、气体流速等），必须完整提取并保留其单位；
        - 不允许省略任何出现于论文中的工艺参数；
        - 若字段值以图表或图片形式出现，也需通过合理推理绑定至相应字段；
        - 抽取结果必须具备可复现实验的能力，即字段值必须完整、真实、精确；
        - 禁止使用模糊词（如“high temperature”或“low pressure”），必须提供带单位的具体数值；
        - 对于范围型数据（如“500 ±10 °C”），必须完整保留；
        - 必须进行精准的单位换算，比如“1 Torr = 133.3 Pa”

      4. 字段扩展规则（Field Expansion Rules）：
        - 仅允许在以下情况中进行字段扩展：
          a. 论文中明确描述了schema未包含但与实验直接相关的工艺信息；
          b. 论文中描述了设备、仪器、控制系统等额外参数，无法归入已有字段，但对实验复现具有重要作用；
        - 所有新字段必须优先嵌入到与其语义最相关的schema模块中（如 Step1/2/3/4/5、equipment_specifications 等）
        - 字段扩展必须具备原文依据，所有新增字段的内容必须能在论文中找到直接文本或图表描述；严禁主观推测、常识补全等形式构造字段
        - 如确有实验内容无法归入现有结构，依据现有schema实现对应扩展
        - 保持层级对齐，避免打平结构

      5. 缺失字段处理规则（Missing Field Handling）：
        - 如果某step在论文中未出现任何字段，仅保留该step的"step_name"，其他未提及的字段不进行输出
        - 若某字段在论文中完全未提及，则从输出中**移除该字段**；
        - 不允许使用以下无效值：null, none, N/A, not specified, unknown 等；
        - 所有输出字段均需具备实际值，否则应当直接省略字段本身。
        - 对于未提及的的流程，宁缺毋滥，不能添加假设字段

      6. 多次沉积步骤处理（Multiple Deposition Steps）：
        - 所有步骤必须以"Step1"、"Step2"等形式出现在process_steps中，处于并列关系，不能嵌套在其他step内部；
        - 对于每一种被沉积材料，必须完整识别该材料在整篇论文中所有与"Film Deposition"相关的沉积行为，包括主工艺段落与任意补充性描述中所涉及的沉积步骤；
        - 若论文中对某材料执行了多次Film Deposition操作，需命名为“Step4_1”、“Step4_2”等，表示不同的沉积轮次；
        - 如果仅执行一次“Film Deposition”，保留“Step4”命名即可，无需编号扩展；
        - 除“Step4”外，Step1/Step2/Step3等步骤**不得嵌套进Step4内部**，必须匹配schema放入合适的位置；
        - 所有沉积步骤必须显式列出，禁止省略任何一次“Film Deposition”。

      7. 多层膜结构（Multilayer Structure）信息描述：
        - 如果论文明确说明沉积了多层膜结构，在所有materials schema之后添加字段"multilayer_design"；
        - 仅当明确存在分层结构时才添加该字段，不能因材料中含多个元素就假定为多层膜；
        - "multilayer_design"字段与每个材料信息处于同一顶层结构；
        - "multilayer_design"字段仅出现一次，必须位于所有material schema之后；
        - "multilayer_design"字段内容必须来源于论文描述，禁止泛化总结或重写原意，确保提取内容真实可溯源、格式规范；
        - 严禁将该字段嵌入到某个"material"实体或"process"内部。

      8. 设备描述（Equipment Description）：
        - 抽取论文中所有关于实验设备的描述，归入字段"equipment_specifications"，包括但不限于真空系统、靶材装置、功率源、气体流量控制系统等；
        - 如果某设备信息可以归入已存在字段，请按模板结构填写；
        - 若无法归入任何已定义字段，请在equipment_specifications中新增字段，并使用专业简洁的英文名称；
        - 新增字段名需保持专业、简洁，并确保内容来源真实、格式规范
        - 该字段只出现一次，始终置于整个JSON结构的最末尾，并与每个材料信息处于同一顶层结构；
        - 禁止将其嵌套入任意材料流程内部。

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有PVD工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种被沉积材料（target_material 与 substrate_material 的组合），检查其是否具备完整、独立的工艺流程建模；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    ''',
    
    "FLUX": '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，依据下列规则对其中的信息进行结构化抽取，并将其归入schema中具体的step。请严格执行每一条指令，不得省略或擅自推断。
      1. 材料提取规则（Material Extraction Rules）：
        - 明确识别论文中所有作为Flux目标生长的晶体材料实体（即被生长晶体）；
        - 如果论文包含多种被生长晶体的Flux流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的Flux工艺流程**，哪怕部分参数（如温度曲线）与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段固定为"material"、"method"、"Steps"，无需添加材料名称前缀（如"[material_name]_process"）。

      2. 结构格式要求（Schema Structure Requirements）：
        - 每个step中必须包含字段"step_name"，且该字段始终放在该step的最前面；
        - 除"step_name"以外的字段请严格遵循模板字段结构（如Step1的precursors、Step3的temperature_profile等），没有必要不添加非模板字段；
        - 所有字段值均为英文表达，禁止添加层级标识、注释或任何非英文内容；
        - 禁止使用中文；
        - 对于step中的子字段（如temperature_profile下的heating_up），请保持其顺序与论文中描述一致，不可重排。
      3. 内容提取规则（Content Extraction Rules）：
        - 所有字段值必须严格基于论文内容提取，不得省略任何能在schema中匹配的字段；
        - 如果论文中提供了实验参数（如温度、时间、摩尔比、纯度等），必须完整提取并保留其单位；
        - 不允许省略任何出现于论文中的工艺参数；
        - 若字段值以图表或图片形式出现，也需通过合理推理绑定至相应字段；
        - 抽取结果必须具备可复现实验的能力，即字段值必须完整、真实、精确；
        - 禁止使用模糊词（如"high temperature"或"low pressure"），必须提供带单位的具体数值（如"1150°C"、"1.5°C/h"）；
        - 对于范围型数据（如"500 ±10 °C"），必须完整保留；
        - 必须进行精准的单位换算（如"1 Torr = 133.3 Pa"，但Flux中较少涉及气压）。
        -引号中的汉字仅作为字段内容的提示词，提取结果时不要出现
      4. 字段扩展规则（Field Expansion Rules）：
        - 仅允许在以下情况中进行字段扩展：
          a. 论文中明确描述了schema未包含但与实验直接相关的工艺信息（如额外处理步骤或设备细节）；
          b. 论文中描述了设备、仪器、控制系统等额外参数，无法归入已有字段，但对实验复现具有重要作用（如炉子类型或坩坩埚埚材质）；
        - 所有新字段必须优先嵌入到与其语义最相关的schema模块中（如Step2的loading_container、Step3的Furnace_type等）；
        - 字段扩展必须具备原文依据，所有新增字段的内容必须能在论文中找到直接文本或图表描述；严禁主观推测、常识补全等形式构造字段；
        - 如确有实验内容无法归入现有结构，依据现有schema实现对应扩展（例如，在Step3中添加新子字段）；
        - 保持层级对齐，避免打平结构。

      5. 缺失字段处理规则（Missing Field Handling）：
        - 如果某step在论文中未出现任何字段，仅保留该step的"step_name"，其他未提及的字段不进行输出；
        - 若某字段在论文中完全未提及，则从输出中**移除该字段**；
        - 不允许使用以下无效值：null, none, N/A, not specified, unknown等；
        - 所有输出字段均需具备实际值，否则应当直接省略字段本身；
        - 对于未提及的流程，宁缺毋滥，不能添加假设字段。

      6. Flux特定步骤处理规则（Flux Step Handling）：
        - 所有步骤必须固定命名为：Step1（Precursor Preparation）、Step2（Loading and Sealing）、Step3（Thermal Processing）、Step4（Crystal Harvesting），并保持线性顺序；
        - 完整识别论文中与原料准备、装填密封、热处理（包括升/降温曲线）、晶体收获相关的所有描述；
        - 如果论文描述了多次独立的热处理循环（如不同温度曲线），需在Step3中以子字段形式扩展（如添加"heating_up_2"），但不得创建新步骤（如Step3_1）；
        - 禁止将任何步骤嵌套进其他step内部（如Step2不能嵌入Step1）。

      7. 设备描述（Equipment Description）：
        - 抽取论文中所有关于实验设备的描述，归入schema中已有字段（如Step2的loading_container、Step3的Furnace_type）；
        - 如果某设备信息无法归入已有字段（如特殊炉子控制系统），请在相关step中新增字段（如Step3中添加"control_system"）；
        - 新增字段名需保持专业、简洁，并确保内容来源真实、格式规范；
        - 设备字段只嵌入在Steps内部，不添加独立顶层字段。

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有Flux工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种被生长材料，检查其是否具备完整、独立的工艺流程建模（包括所有Steps）；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    ''',
    
    "CVT":  '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，依据下列规则对其中的信息进行结构化抽取，并将其归入schema中具体的step。请严格执行每一条指令，不得省略或擅自推断。
      1. 材料提取规则（Material Extraction Rules）：
        - 明确识别论文中所有作为CVT目标材料的实体（即生长材料）；
        - 如果论文包含多种生长材料的CVT流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的CVT工艺流程**，哪怕部分参数与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段命名为："[material_name]_process"，其中material_name为该材料在论文中出现的**完整英文名称**。
      2. 结构格式要求（Schema Structure Requirements）：
        - 每个step中必须包含字段“step_name”，且该字段始终放在该step的最前面；
        - 除“step_name”以外的字段请严格遵循模板字段结构，没有必要不添加非模板字段；
        - 所有字段值均为英文表达，禁止添加层级标识和注释等内容
        - ！！禁止使用中文！！；
        - 对于step中的子字段，请保持其顺序与论文中描述一致，不可重排
      3. 内容提取规则（Content Extraction Rules）：
        - 所有字段值必须严格基于论文内容提取，不得省略任何能在schema中匹配的字段；
        - 如果论文中提供了实验参数（如温度、气压、功率、气体流速等），必须完整提取并保留其单位；
        - 不允许省略任何出现于论文中的工艺参数；
        - 若字段值以图表或图片形式出现，也需通过合理推理绑定至相应字段；
        - 抽取结果必须具备可复现实验的能力，即字段值必须完整、真实、精确；
        - 禁止使用模糊词（如“high temperature”或“low pressure”），必须提供带单位的具体数值；
        - 对于范围型数据（如“500 ±10 °C”），必须完整保留；
        - 必须进行精准的单位换算，比如“1 Torr = 133.3 Pa”
      4. 字段扩展规则（Field Expansion Rules）：
        - 仅允许在以下情况中进行字段扩展：
          a. 论文中明确描述了schema未包含但与实验直接相关的工艺信息；
          b. 论文中描述了设备、仪器、控制系统等额外参数，无法归入已有字段，但对实验复现具有重要作用；
        - 所有新字段必须优先嵌入到与其语义最相关的schema模块中（如 Step1/2/3/4/5/6/7、equipment_specifications 等）
        - 字段扩展必须具备原文依据，所有新增字段的内容必须能在论文中找到直接文本或图表描述；严禁主观推测、常识补全等形式构造字段
        - 如确有实验内容无法归入现有结构，依据现有schema实现对应扩展
        - 保持层级对齐，避免打平结构
      5. 缺失字段处理规则（Missing Field Handling）：
        - 如果某step在论文中未出现任何字段，仅保留该step的"step_name"，其他未提及的字段不进行输出
        - 若某字段在论文中完全未提及，则从输出中**移除该字段**；
        - 不允许使用以下无效值：null, none, N/A, not specified, unknown 等；
        - 所有输出字段均需具备实际值，否则应当直接省略字段本身。
        - 对于未提及的的流程，宁缺毋滥，不能添加假设字段
      6. 多次生长步骤处理（Multiple Growth Steps）：
        - 所有步骤必须以"Step1"、"Step2"等形式出现在process_steps中，处于并列关系，不能嵌套在其他step内部；
        - 对于每一种生长材料，必须完整识别该材料在整篇论文中所有与"Crystal Growth"相关的生长行为，包括主工艺段落与任意补充性描述中所涉及的生长步骤；
        - 若论文中对某材料执行了多次Crystal Growth操作，需命名为“Step6_1”、“Step6_2”等，表示不同的生长轮次（基于模板Step6）；
        - 如果仅执行一次“Crystal Growth”，保留“Step6”命名即可，无需编号扩展；
        - 除“Step6”外，Step1/Step2/Step3等步骤**不得嵌套进Step6内部**，必须匹配schema放入合适的位置；
        - 所有生长步骤必须显式列出，禁止省略任何一次“Crystal Growth”。
      7. 多层结构信息描述（Multilayer Structure Description）：
        - 如果论文明确说明存在多层结构（如分层生长），在所有materials schema之后添加字段"multilayer_design"；
        - 仅当明确存在分层结构时才添加该字段，不能因材料中含多个元素就假定为多层结构；
        - "multilayer_design"字段与每个材料信息处于同一顶层结构；
        - "multilayer_design"字段仅出现一次，必须位于所有material schema之后；
        - "multilayer_design"字段内容必须来源于论文描述，禁止泛化总结或重写原意，确保提取内容真实可溯源、格式规范；
        - 严禁将该字段嵌入到某个"material"实体或"process"内部。

      8. 设备描述（Equipment Description）：
        - 抽取论文中所有关于实验设备的描述，归入字段"equipment_specifications"，包括但不限于真空系统、温区装置、传输剂处理系统、气体控制系统等；
        - 如果某设备信息可以归入已存在字段，请按模板结构填写；
        - 若无法归入任何已定义字段，请在equipment_specifications中新增字段，并使用专业简洁的英文名称；
        - 新增字段名需保持专业、简洁，并确保内容来源真实、格式规范
        - 该字段只出现一次，始终置于整个JSON结构的最末尾，并与每个材料信息处于同一顶层结构；
        - 禁止将其嵌套入任意材料流程内部。

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有CVT工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种生长材料（目标生长材料），检查其是否具备完整、独立的工艺流程建模；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    ''',
    
    "CVD": '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，并将其归入schema中具体的step。请严格遵循以下规则：
      0. 材料提取规则：
        - 明确识别论文中所有作为CVD目标材料的实体（即被沉积材料）；
        - 如果论文包含多种被沉积材料的CVD流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的CVD工艺流程**，哪怕部分参数与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段命名为："[material_name]_process"，其中material_name为该材料在论文中出现的**完整英文名称**。
      1. 严格遵循结构：
        -每个step必须保留"step_name"字段
        -格式必须与模板保持一致，且模板中不包含的内容不进行提取
        -提取结果不包含层级标识和注释等内容，内容必须全部为英文表达
      2. 内容提取规则：
        -如果论文中不含有具体的CVD材料制备工艺，则停止提取凝练
        -必须完全基于论文内容凝练字段
        -对于schema中能够匹配的字段，必须严格按照模板进行提取
        -能够提取出论文中实验部分的全部关键信息没有遗漏
      3. 空缺处理：
        -对于论文未提及的步骤，只保留`"step_name"字段，其他未提及的字段不进行输出
        -（必须严格遵守）不填写类似null，not specified等无意义的字段
        -对于未提及的的流程，宁缺毋滥，不能添加假设字段
      4. 多次沉积处理流程：
        -对于每种被沉积材料，识别所有该材料在论文中的`Film Deposition`实例
        -对每次沉积必须独立创建子步骤，注意Step4中的Film Deposition可能会执行多次，要确保模板覆盖文章中的所有步骤
        -只有对于多次执行的情况，为每个Film Deposition步骤添加编号，命名方式为Step4_N
        -对于单次执行的情况，不必添加编号，命名方式保留为Step4
      5. 设备说明处理：
        -设备说明必须严格按照论文明确的内容进行提取，不能添加假设字段
        -如果论文中存在"equipment_specifications"下不存在的字段，判断是否能够放入当前已有的字段中，如果能够放入请按照正确的层级格式进行排列
        -如果新增字段无法放入当前已有的字段中，请按照格式新建字段，并进行完整补充
        -"equipment_specifications"在schema中只会出现一次，放在最后并且与"material"层级同级
      6. 将更新后的schema严格输出为json格式，确保输出结果只为json并检查不存在格式错误

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有CVD工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种被生长材料，检查其是否具备完整、独立的工艺流程建模（包括所有Steps）；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    ''',
    
    "MOCVD": '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，并将其归入schema中具体的step。请严格遵循以下规则：
      0. 材料提取规则：
        - 明确识别论文中所有作为MOCVD目标材料的实体（即被沉积材料）；
        - 如果论文包含多种被沉积材料的MOCVD流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的MOCVD工艺流程**，哪怕部分参数与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段命名为："[material_name]_process"，其中material_name为该材料在论文中出现的**完整英文名称**。
      1. 严格遵循结构：
        -每个step必须保留"step_name"字段
        -格式必须与模板保持一致，且模板中不包含的内容不进行提取
        -提取结果不包含层级标识和注释等内容，内容必须全部为英文表达
      2. 内容提取规则：
        -如果论文中不含有具体的MOCVD材料制备工艺，则停止提取凝练
        -必须完全基于论文内容凝练字段
        -对于schema中能够匹配的字段，必须严格按照模板进行提取
        -能够提取出论文中实验部分的全部关键信息没有遗漏
      3. 空缺处理：
        -对于论文未提及的步骤，只保留`"step_name"字段，其他未提及的字段不进行输出
        -（必须严格遵守）不填写类似null，not specified等无意义的字段
        -对于未提及的的流程，宁缺毋滥，不能添加假设字段
      4. 多次沉积处理流程：
        -对于每种被沉积材料，识别所有该材料在论文中的`Film Deposition`实例
        -对每次沉积必须独立创建子步骤，注意Step4中的Film Deposition可能会执行多次，要确保模板覆盖文章中的所有步骤
        -只有对于多次执行的情况，为每个Film Deposition步骤添加编号，命名方式为Step4_N
        -对于单次执行的情况，不必添加编号，命名方式保留为Step4
      5. 设备说明处理：
        -设备说明必须严格按照论文明确的内容进行提取，不能添加假设字段
        -如果论文中存在"equipment_specifications"下不存在的字段，判断是否能够放入当前已有的字段中，如果能够放入请按照正确的层级格式进行排列
        -如果新增字段无法放入当前已有的字段中，请按照格式新建字段，并进行完整补充
        -"equipment_specifications"在schema中只会出现一次，放在最后并且与"material"层级同级
      6. 将更新后的schema严格输出为json格式，确保输出结果只为json并检查不存在格式错误

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有MOCVD工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种被生长材料，检查其是否具备完整、独立的工艺流程建模（包括所有Steps）；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    ''',
    
    "MBE": '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，并将其归入schema中具体的step。请严格遵循以下规则：
      0. 材料提取规则：
        - 明确识别论文中所有作为MBE目标材料的实体（即被沉积材料）；
        - 如果论文包含多种被沉积材料的MBE流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的MBE工艺流程**，哪怕部分参数与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段命名为："[material_name]_process"，其中material_name为该材料在论文中出现的**完整英文名称**。
      1. 严格遵循结构：
        -每个step必须保留"step_name"字段
        -格式必须与模板保持一致，且模板中不包含的内容不进行提取
        -提取结果不包含层级标识和注释等内容，内容必须全部为英文表达
      2. 内容提取规则：
        -如果论文中不含有具体的MBE材料制备工艺，则停止提取凝练
        -必须完全基于论文内容凝练字段
        -对于schema中能够匹配的字段，必须严格按照模板进行提取
        -能够提取出论文中实验部分的全部关键信息没有遗漏
      3. 空缺处理：
        -对于论文未提及的步骤，只保留`"step_name"字段，其他未提及的字段不进行输出
        -（必须严格遵守）不填写类似null，not specified等无意义的字段
        -对于未提及的的流程，宁缺毋滥，不能添加假设字段
      4. 多次沉积处理流程：
        -沉积部分（对应Film Deposition）是模板抽取的重点部分，请严格遵守处理规则，尽最大可能提高这部分抽取内容的准确率和覆盖率
        -对于每种被沉积材料，识别所有该材料在论文中的`Film Deposition`实例
        -对每次沉积必须独立创建子步骤，注意Step4中的Film Deposition可能会执行多次，要确保模板覆盖文章中的所有步骤
        -只有对于多次执行的情况，为每个Film Deposition步骤添加编号，命名方式为Step4_N
        -对于单次执行的情况，不必添加编号，命名方式保留为Step4
      5. 设备说明处理：
        -设备说明必须严格按照论文明确的内容进行提取，不能添加假设字段
        -如果论文中存在"equipment_specifications"下不存在的字段，判断是否能够放入当前已有的字段中，如果能够放入请按照正确的层级格式进行排列
        -如果新增字段无法放入当前已有的字段中，请按照格式新建字段，并进行完整补充
        -"equipment_specifications"在schema中只会出现一次，放在最后并且与"material"层级同级
      6. 将更新后的schema严格输出为json格式，确保输出结果只为json并检查不存在格式错误

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有MBE工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种被生长材料，检查其是否具备完整、独立的工艺流程建模（包括所有Steps）；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    ''',
    
    "ALD": '''
    B. 仔细阅读论文，将论文中提到的工艺步骤凝练成简洁标准的工艺流程字段，并将其归入schema中具体的step。请严格遵循以下规则：
      0. 材料提取规则：
        - 明确识别论文中所有作为ALD目标材料的实体（即被沉积材料）；
        - 如果论文包含多种被沉积材料的ALD流程，你必须**为每种材料分别创建一个独立的schema**；
        - 每个材料的schema必须完整描述其**独立的ALD工艺流程**，哪怕部分参数与其他材料重复；
        - 不能将多个材料的工艺流程混合在一个schema中；
        - 如果不同材料的某些参数完全一致，请在每个材料对应的schema中分别保留该参数，这些字段和对应的参数值在每个独立schema中必须保持完全一致;
        - 每个schema的顶级字段命名为："[material_name]_process"，其中material_name为该材料在论文中出现的**完整英文名称**。
      1. 严格遵循结构：
        -每个step必须保留"step_name"字段
        -格式必须与模板保持一致，且模板中不包含的内容不进行提取
        -提取结果不包含层级标识和注释等内容，内容必须全部为英文表达
      2. 内容提取规则：
        -如果论文中不含有具体的ALD材料制备工艺，则停止提取凝练
        -必须完全基于论文内容凝练字段
        -对于schema中能够匹配的字段，必须严格按照模板进行提取
        -能够提取出论文中实验部分的全部关键信息没有遗漏
      3. 空缺处理：
        -对于论文未提及的步骤，只保留`"step_name"字段，其他未提及的字段不进行输出
        -（必须严格遵守）不填写类似null，not specified等无意义的字段
        -对于未提及的的流程，宁缺毋滥，不能添加假设字段
      4. 多次沉积处理流程：
        -沉积部分（对应Film Deposition）是模板抽取的重点部分，请严格遵守处理规则，尽最大可能提高这部分抽取内容的准确率和覆盖率
        -对于每种被沉积材料，识别所有该材料在论文中的`Film Deposition`实例
        -对每次沉积必须独立创建子步骤，注意Step3中的Film Deposition可能会执行多次，要确保模板覆盖文章中的所有步骤
        -只有对于多次执行的情况，为每个Film Deposition步骤添加编号，命名方式为Step3_N
        -对于单次执行的情况，不必添加编号，命名方式保留为Step3
      5. 设备说明处理：
        -设备说明必须严格按照论文明确的内容进行提取，不能添加假设字段
        -如果论文中存在"equipment_specifications"下不存在的字段，判断是否能够放入当前已有的字段中，如果能够放入请按照正确的层级格式进行排列
        -如果新增字段无法放入当前已有的字段中，请按照格式新建字段，并进行完整补充
        -"equipment_specifications"在schema中只会出现一次，放在最后并且与"material"层级同级
      6. 将更新后的schema严格输出为json格式，确保输出结果只为json并检查不存在格式错误

    C. 在完成整篇论文的信息抽取后，必须对抽取得到的内容进行严格校验，确保可复现实验流程，执行以下检查流程：
      1. 检查是否有遗漏字段，若存在，应立即回溯原文补全；
      2. 检查所有ALD工艺相关内容是否已被映射至schema字段中；
      3. 针对每一种被生长材料，检查其是否具备完整、独立的工艺流程建模（包括所有Steps）；
      4. 检查所有字段格式、层级、字段值是否满足可复现实验流程的要求。

    D. 字段值有效性强制校验（Field Value Validity）：
      - 不得出现以下无效值："null", "none", "N/A", "not specified", "not mentioned", "unknown"；
      - 若字段值为上述之一，必须从输出中彻底移除该字段；
      - 禁止使用这些无效值的任何形式（如大小写变体、缩写等）；
      - ！必须在输出JSON前执行该校验！

    E. 输出格式要求（Output Requirements）：
      - 最终输出必须为严格符合语法规范的JSON结构；
      - 仅输出JSON，不添加任何额外文本、注释或提示信息；
      - 保证JSON中无语法错误，字段层级清晰，格式规范；
      - 在生成JSON前请确保所有字段已通过有效性校验。
    '''
}

# 提取文本信息
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def craft_extract(extracted_text, craft_class, client):
    system_prompt = f"""
        您是一位顶级的材料制备工程师，熟悉{craft_class}材料制备方法
    """
    with open(f"craft_schema/{craft_class}.json", "r", encoding="utf-8") as f:
        json_template = f.read()
        # template_dict = json.load(f)
        # json_template = json.dumps(template_dict, indent=2, ensure_ascii=False)

    extract_prompt = f"""
        你是一位顶级的材料制备工程师，熟悉{craft_class}材料制备方法，从{craft_class}方法材料制备的论文中抽取目标材料信息字段填入material类中；抽取工艺字段，填入已有{craft_class}工艺的schema中process中的step类；抽取形成一个用于格式化提取{craft_class}工艺的详细schema。
        如下是论文的原文信息
        {extracted_text}
        A. 工艺提取模板基于如下的材料schema：
        {json_template}
        {craft_prompt_dict[craft_class]}
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": extract_prompt}]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                'type': 'json_object'
            }
        )
        extract_json = response.choices[0].message.content
        return extract_json
    
    except BadRequestError as e:
        # 捕获400错误并跳过当前PDF
        print(f"工艺提取输入过长错误: {e}")
        print(f"跳过当前PDF文件")
        return None
  
# 提取工艺并保存为JSON文件
def extract_single_pdf(pdf_path, folder_process_path, api_key):
    """单个PDF文件抽取"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com")
        
        # 提取PDF文本
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            print(f"警告：{os.path.basename(pdf_path)} 未提取到文本内容")
            return False
        
        #找出工艺类别名称
        craft_class = os.path.basename(os.path.dirname(pdf_path))
        craft_rlt = craft_extract(extracted_text, craft_class, client)
        if craft_rlt is None:
            return False
        else:
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            json_path = os.path.join(folder_process_path, craft_class, f"{pdf_filename}.json")
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            with open(json_path, "w", encoding="utf-8") as json_file:
                json_file.write(craft_rlt)
            return True
        
    except Exception as e:
        print(f"处理文件 {os.path.basename(pdf_path)} 时发生错误: {e}")
        return False    

def collect_all_pdfs_for_extract(folder_path):
    pdf_paths = []
    for root, dirs, files in os.walk(folder_path):
        # 跳过根目录下的 other 和 non-process 文件夹
        if 'other' in root.split(os.path.sep) or 'non-process' in root.split(os.path.sep):
            continue
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_paths.append(pdf_path)
    return pdf_paths

if __name__ == '__main__':
    start_time = time.perf_counter()
    api_key="sk-e4225d0fac984681a976a83c6c2997c7"

    folder_path = f'../multi_material_process' #待抽取文件夹
    folder_extract_path = folder_path + '_extract_v3' #抽取后文件夹
    os.makedirs(folder_extract_path, exist_ok=True)

    pdf_paths = collect_all_pdfs_for_extract(folder_path)
    total_files = len(pdf_paths)
    print(f"共有{total_files}个PDF文件待处理")
    if total_files == 0:
        print("没有找到PDF文件")
        exit()
    
    # extract_single_pdf(pdf_paths[0], folder_extract_path, api_key)
        
    # 多进程
    max_processes = multiprocessing.cpu_count()
    # max_processes = 48
    print(f"使用 {max_processes} 个进程进行处理")
    process_func = partial(
        extract_single_pdf,
        folder_process_path=folder_extract_path,
        api_key=api_key
    )

    with multiprocessing.Pool(processes=max_processes) as pool:

        results = pool.imap_unordered(process_func, pdf_paths)
        
        # 统计成功和失败的数量
        success_count = 0
        for i, result in enumerate(results, 1):
            if result:
                success_count += 1

            if i % 10 == 0 or i == total_files:
                print(f"处理进度: {i}/{total_files} ({success_count} 个成功)")
    
    # 输出最终统计信息
    end_time = time.perf_counter()
    print(f"\n处理完成！总文件数: {total_files}, 成功: {success_count}, 失败: {total_files - success_count}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每个文件处理时间: {(end_time - start_time)/total_files:.2f} 秒")