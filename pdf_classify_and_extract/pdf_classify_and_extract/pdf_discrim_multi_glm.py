import json
from openai import OpenAI, BadRequestError
from pypdf import PdfReader
import os
import time
from zai import ZhipuAiClient
import shutil
import multiprocessing
from functools import partial

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

# 提取文本信息
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def pdf_category_discrimination(pdf_constent, client):
    
    system_prompt = """
      您是材料科学领域的专业研究助理，专注于判别材料科学论文的工艺类别。您将获得一篇材料科学论文的文本，请从以下工艺类别中判别该论文所涉及的主要工艺：
      - PVD: 物理气相沉积 (Physical Vapor Deposition)
      - FLUX: 熔剂法 (Flux Method)
      - CVT: 化学气相传输 (Chemical Vapor Transport)
      - CVD: 化学气相沉积 (Chemical Vapor Deposition)
      - MOCVD: 金属有机化学气相沉积 (Metal-Organic Chemical Vapor Deposition)
      - MBE: 分子束外延 (Molecular Beam Epitaxy)
      - ALD: 原子层沉积 (Atomic Layer Deposition)
    """
    discrimination_prompt = f"""
      Only answer with one of the following categories:
      "PVD", "FLUX", "CVT", "CVD", "MOCVD", "MBE", "ALD", "other", or "non-process"
      如下是原文信息
      {pdf_constent}
      请严格判断该文章是否包含 **某一工艺的完整制作流程**，严格遵守以下规则：
        - 如果涉及到上述七类工艺之一，请输出对应的缩写（如 "PVD"）。
        - 如果涉及的工艺不是这七类，而是其他工艺或进阶/变种方法（如 HS-PVD 等），请输出 "other"。
        - 如果文章根本不涉及任何工艺流程（例如纯理论研究、性能测试、计算模拟等），请输出 "non-process"。
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": discrimination_prompt}]

    try:
        #GLM
        response = client.chat.completions.create(
          model="glm-4-flash",
          messages=messages,
          temperature=0.01
        )

        #deepseek
        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=messages,
        #     stream=False
        # )

        discrimination_rlt = response.choices[0].message.content
        if discrimination_rlt in craft_dict.keys():
            return discrimination_rlt
            
    except BadRequestError as e:
        # 捕获400错误并跳过当前PDF
        print(f"分类输入过长错误: {e}")
        print(f"跳过当前PDF文件")
        return 'other'
        
    except Exception as e:
        # 其他错误正常处理
        print(f"分类错误: {e}")
        return 'other'

def process_single_pdf(pdf_path, folder_process_path, api_key):
    """单个PDF文件的处理"""
    try:
        client = ZhipuAiClient(api_key=api_key)
        # client = OpenAI(
        #     api_key=api_key,
        #     base_url="https://api.deepseek.com")
        
        # 提取PDF文本
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            print(f"警告：{os.path.basename(pdf_path)} 未提取到文本内容")
            return False
        
        # 分类判断
        pdf_class = pdf_category_discrimination(extracted_text, client)
        
        # 如果分类在字典中，则复制文件
        if pdf_class in craft_dict.keys():
            # 创建目标文件夹
            path_process = os.path.join(folder_process_path, pdf_class)
            os.makedirs(path_process, exist_ok=True)

            # 复制文件
            target_path = os.path.join(path_process, os.path.basename(pdf_path))
            shutil.copy2(pdf_path, target_path)
            print(f"已处理并复制: {os.path.basename(pdf_path)} -> {pdf_class}")
            
        elif pdf_class == 'other':
            # 创建other文件夹
            path_process = os.path.join(folder_process_path, 'other')
            os.makedirs(path_process, exist_ok=True)
            
            # 复制文件
            target_path = os.path.join(path_process, os.path.basename(pdf_path))
            shutil.copy2(pdf_path, target_path)
            print(f"已处理并复制: {os.path.basename(pdf_path)} -> other")
        
        else:
            # 创建non-process文件夹
            path_process = os.path.join(folder_process_path, 'non-process')
            os.makedirs(path_process, exist_ok=True)
            
            # 复制文件
            target_path = os.path.join(path_process, os.path.basename(pdf_path))
            shutil.copy2(pdf_path, target_path)
            print(f"已处理并复制: {os.path.basename(pdf_path)} -> non-process")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {os.path.basename(pdf_path)} 时发生错误: {e}")
        return False

def collect_all_pdfs(folder_path):
    pdf_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_paths.append(pdf_path)
    return pdf_paths

if __name__ == '__main__':
    start_time = time.perf_counter()

    #pdf存储路径
    folder_path = f'multi_material' #待分类文件夹
    folder_process_path = folder_path + '_process' #分类后文件夹

    os.makedirs(folder_process_path, exist_ok=True)

    api_key="2ad72e41beb4c25dc4dcf384779b8556.9xCBJwxxZNla7d7Q"
    # api_key = "sk-e4225d0fac984681a976a83c6c2997c7" #deepseek

    pdf_paths = collect_all_pdfs(folder_path)
    total_files = len(pdf_paths)
    print(f"共有{total_files}个PDF文件待处理")
    if total_files == 0:
        print("没有找到PDF文件")
        exit()

    # 配置多进程
    max_processes = multiprocessing.cpu_count()
    # max_processes = 48
    print(f"使用 {max_processes} 个进程进行处理")

    process_func = partial(
        process_single_pdf,
        folder_process_path=folder_process_path,
        api_key=api_key
    )
    with multiprocessing.Pool(processes=max_processes) as pool:
        # 使用imap_unordered可以实时获取处理结果
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