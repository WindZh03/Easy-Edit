# Easy-Edit: 使用LLM提高图像编辑模型的性能

当用户输入的编辑指令为"把第三个红色的苹果变成篮球"时，Instruct-Pix2Pix 模型无法理解指令中的"第三个"和"红色"属性信息，因此无法正确地定位到目标对象。

本项目利用 LLM(Qwen) + Grounded-SAM 实现更精细的目标定位与掩码提取，一定程度上提高了 Instruct-Pix2Pix 模型针对这类编辑指令的性能。


<img src="./src/img1.png" alt="预览图" width="400" style=" height: auto;">


## 工作原理

Easy-Edit 的执行逻辑为：

1.  **理解指令**：使用 LLM 理解用户的指令，并解析出待编辑对象及其属性信息（目前只实现了三种属性信息，分别是相对位置、相对大小、颜色信息）。
2.  **目标定位**：使用 Grounded-SAM 精确定位到待编辑对象，并提取其掩码。
3.  **执行编辑**：根据解析的指令，使用预训练的 Instruct-Pix2Pix 进行图像编辑。

## 安装说明

### 1. 设置API密钥

在 `utils.py` 文件中设置您的 LLM API 密钥和阿里云 API 密钥。

*   **DashScope API Key**: [点击此处](https://help.aliyun.com/zh/model-studio/get-api-key?spm=a2c4g.11186623.help-menu-2400256.d_2_0_0.4da31c90LRW7pP)获取您的 `api_key`。
*   **Alibaba Cloud API Keys**: 用于访问阿里云相关服务。

```python
# 编辑 utils.py 文件
dashscope.api_key=""
os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = ''
os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = ''
```

### 2. 安装依赖项

使用 pip 安装所有必需的 Python 库。

```bash
pip install -r requirements.txt
```

### 3. 下载NLP模型

下载 `spaCy` 的英语语言模型，用于自然语言处理。

```bash
python -m spacy download en_core_web_sm
```

### 4. 安装GroundingDINO

GroundingDINO 用于根据文本描述在图像中定位对象。

```bash
# 设置您的 CUDA 安装路径
export CUDA_HOME=/path/to/your/cuda/installation

# 克隆仓库并安装
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
python setup.py install

# 创建目录并下载预训练权重
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../
```

如果您在安装过程中遇到任何问题，请参考官方的 [GroundingDINO 项目仓库](https://github.com/IDEA-Research/GroundingDINO)。

## 如何使用

本项目提供了两种使用方式：

### 1. Jupyter Notebook

您可以使用提供的 Jupyter Notebook 来进行交互式开发和测试：

```bash
# 打开并运行
easy_edit.ipynb
```

### 2. Streamlit 交互式应用

此外，项目还包含一个使用 Streamlit 构建的交互式 Web 应用。您可以通过以下命令来启动它：

```bash
streamlit run streamlit_edit.py
```
<img src="./src/img2.png" alt="预览图" width="600" height="400" style="max-width: 100%; height: auto;">

## 参考

本项目参考了以下项目：
*   [Grounded-Instruct-Pix2Pix](https://github.com/arthur-71/Grounded-Instruct-Pix2Pix) (本项目的基础)
*   [Instruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
