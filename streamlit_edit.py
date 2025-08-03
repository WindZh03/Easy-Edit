import streamlit as st
from PIL import Image
import time
import math
from PIL import Image, ImageOps
from easy_edit import process_image
import os
import utils
import torch

# 设置页面配置 - 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="图像编辑系统",
    page_icon="🎨",
    layout="wide"
)

# 设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_pil_image(image_path, resolution=512):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return image



# 调整图像到固定显示尺寸，保持比例
def resize_image_for_display(image, max_width=256, max_height=300):
    """调整图像尺寸以适合显示，保持原始比例"""
    if image is None:
        return None
    
    # 创建图像副本以避免修改原始图像
    img_copy = image.copy()
    width, height = img_copy.size
    
    # 计算缩放比例以适应最大尺寸
    scale_factor = min(max_width/width, max_height/height)
    
    if scale_factor < 1:  # 仅当图像过大时才缩小
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img_copy




# 自定义CSS样式 - 修复标题显示问题
st.markdown("""
<style>
    /* 主题颜色 */
    :root {
        --primary-color: #4f46e5; /* 主色调 - 靛蓝色 */
        --secondary-color: #f3f4f6; /* 次要色调 - 轻灰色 */
        --accent-color: #8b5cf6; /* 强调色 - 紫色 */
        --text-color: #1e293b; /* 文字颜色 - 深灰蓝色 */
        --border-color: #e2e8f0; /* 边框颜色 - 浅灰色 */
    }
    
    /* 页面背景 */
    .main {
        background-color: #f8fafc;
        padding: 1.5rem;
    }
    
    /* 修复标题样式 */
    h1 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
        padding-bottom: 1rem !important;
        border-bottom: 1px solid var(--border-color) !important;
        margin-bottom: 1.5rem !important;
        font-size: 2rem !important;
        line-height: 1.3 !important;
        overflow: visible !important;
        white-space: normal !important;
    }
    
    h2, h3 {
        color: var(--text-color) !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
    }
    
    /* 卡片样式 */
    div.stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #4338ca; /* 深蓝色 - 悬停状态 */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* 确保两列等宽 */
    [data-testid="column"] {
        width: 50% !important;
        flex: 1 1 calc(50% - 1rem) !important;
        min-width: calc(50% - 1rem) !important;
    }
    
    /* 容器样式 */
    .block-container {
        max-width: 95% !important;
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* 文本区域样式 */
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        padding: 0.75rem;
    }
    
    /* 滑块样式 */
    .stSlider div[data-baseweb="slider"] {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* 文件上传组件样式 */
    .stFileUploader div[data-testid="stFileUploader"] {
        border-radius: 0.5rem;
        border: 2px dashed var(--border-color);
        padding: 1rem;
    }
    
    /* 图片容器样式 */
    .stImage {
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* 提示文本样式 */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* 进度指示器 */
    .stProgress div {
        border-radius: 0.5rem !important;
    }
    
    /* 确保标题容器有足够空间 */
    .stMarkdown {
        width: 100% !important;
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

 
with st.container():
    # 标题区
    st.title("🎨 基于自然语言提示的图像编辑系统")
    # st.subheader("基于Instruct-Pix2Pix的智能图像编辑工具")

# 创建两列布局
col1, col2 = st.columns([1, 1])

# 左侧面板 - 输入区域
with col1:
    st.markdown("### 📸 上传图像")
    
    # 图像上传区域
    input_image = st.file_uploader("选择一张图片进行编辑", type=["png", "jpg", "jpeg"], key="input_image_uploader")
    if input_image:
        input_image = Image.open(input_image)
        
        # 打开并处理图像
        # original_img = Image.open(input_image)
        # 调整为固定尺寸显示
        display_img = resize_image_for_display(input_image, 256, 300)
        
        st.image(display_img, caption="原始图像", use_container_width=False)
    
    
    st.markdown("### 🖼️ 编辑结果")
    # 输出图像区域
    output_image = st.empty()
    

# 右侧面板 - 输出区域
with col2:
    
    st.markdown("### 🤖 模型选择")
    model_choice = st.radio(
        "选择要使用的编辑模型",
        options=["My method","Instruct-Pix2Pix"],
        index=0,
        help="不同模型可能有不同的编辑效果"
    )
    
    # 编辑指令区域
    st.markdown("### 📝 编辑指令")
    edit_text = st.text_area(
        "描述您想要对图像做的修改（建议使用英文）",
        placeholder="输入您的编辑要求：\n- 增加图像亮度\n- 添加暖色调滤镜\n- 将背景改为冬季场景",
        height=120
    )
    
    
    # 按钮行
    col1_1, col1_2, col1_3 = st.columns(3)
    with col1_1:
        edit_btn = st.button("✨ 开始编辑", use_container_width=True, key="edit_button")
    with col1_2:
        clear_btn = st.button("🗑️ 清除", use_container_width=True, key="clear_button")
    with col1_3:
        show_btn = st.button("🖼️ 显示结果", use_container_width=True, key="show_button")
    
    # 高级参数设置
    with st.expander("⚙️ 高级参数设置", expanded=True):
        strength_slider = st.slider(
            "图像保真度 (Image CFG Scale)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help="控制输出图像与输入图像的相似度，值越高输出越接近原图"
        )
        
        noise_slider = st.slider(
            "文本引导强度 (Text CFG Scale)",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.5,
            help="控制对文本指令的遵循程度，值越高越严格遵循编辑指令"
        )
        
        random_seed = st.slider(
            "随机种子 (Random Seed)",
            min_value=0,
            max_value=50,
            value=42,
            step=1,
            help="控制随机性，相同种子会产生相似结果"
        )

# 功能实现部分
def edit_image(image, edit_prompt, strength_slider, noise_slider, random_seed, model_choice):
    if image is None:
        return None
    
    image_path = utils.save_image(image)
    image = load_pil_image(image_path)
    if model_choice == "Instruct-Pix2Pix":
        use_pix = True
    else:
        use_pix = False
    try:
        processed = process_image(
            image, 
            edit_prompt,
            image_guidance_scale=strength_slider,
            text_guidance_scale=noise_slider,
            seed=random_seed,
            use_pix=use_pix
        )
        
        if processed is None:
            st.error("处理失败，未返回有效图像")
            return None
            
        torch.cuda.empty_cache()
        utils.save_image(processed)
        st.session_state.result_image = processed
        return processed
        
    except Exception as e:
        st.error(f"处理失败：{str(e)}")
        torch.cuda.empty_cache()
        return None

def clear_outputs():
    if 'result_image' in st.session_state:
        del st.session_state.result_image
    st.session_state.clear()

def show_image():
    try:
        edited_image = load_pil_image('saved_images/edited_image.png')
        if edited_image is not None:
            # 保存到会话状态
            st.session_state.result_image = edited_image
            return edited_image
        else:
            st.warning("没有可显示的编辑图像")
            return None
    except Exception as e:
        st.error(f"显示图像失败：{str(e)}")
        return None

# 处理按钮点击事件
if edit_btn and input_image and edit_text:
    with st.spinner("🔄 正在处理图像..."):
        result = edit_image(input_image, edit_text, strength_slider, noise_slider, random_seed,model_choice)
        if result:
            result = resize_image_for_display(result, 256, 300)
            output_image.image(result, caption="编辑后的图像", use_container_width=False)

if clear_btn:
    clear_outputs()
    st.experimental_rerun()

if show_btn:
    result = show_image()
    if result:
        output_image.image(result, caption="编辑后的图像", use_container_width=True)

# 添加页脚
st.markdown("---")
st.markdown("### 📖 使用说明")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **基本步骤:**
    1. 上传一张需要编辑的图片
    2. 输入描述编辑效果的文字指令
    3. 调整参数以获得最佳效果
    4. 点击"开始编辑"按钮
    """)

with col_b:
    st.markdown("""
    **参数说明:**
    - **图像保真度**: 控制输出与原图的相似度
    - **文本引导强度**: 控制对编辑指令的遵循程度
    - **随机种子**: 固定种子可重现相同的编辑效果
    """)