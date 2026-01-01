import streamlit as st
from config.settings import CONFIG
from utils.utils import load_model
from utils.features import batch_analyze_files, derive_acd_standard

def main():
    # 页面配置
    st.set_page_config(
        page_title="ACd批量分析与标准推导",
        page_icon="📊",
        layout="wide"
    )
    st.title("📊 ACd批量分析 + 动态有效态保护标准推导")
    st.divider()
    
    # 加载模型
    with st.spinner("🔧 加载ACd预测模型..."):
        try:
            model = load_model()
            st.success("✅ 模型加载成功（XGBoost，对数尺度训练）")
        except Exception as e:
            st.error(f"❌ 模型加载失败：{str(e)}")
            return
    
    # 批量文件上传与分析
    st.subheader("🔹 步骤1：批量上传样本数据")
    uploaded_files = st.file_uploader(
        "选择CSV/XLSX文件（支持多文件）",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help=f"文件需包含列：{', '.join(CONFIG['FEATURE_COLS'] + [CONFIG['TARGET_COL']])}"
    )
    
    data_stats = None
    r2_log_scale = None
    
    if uploaded_files:
        st.divider()
        st.subheader("🔹 步骤2：批量数据分析结果")
        data_stats, r2_log_scale = batch_analyze_files(uploaded_files, model)
    
    # ACd标准推导
    st.divider()
    st.subheader("🔹 步骤3：ACd有效态保护标准推导（3D预测）")
    if data_stats is None or r2_log_scale is None:
        st.info("💡 请先完成步骤1-2的批量数据上传与分析，系统将基于上传数据的变量范围推导ACd标准")
    else:
        with st.expander("⚙️ 微调变量范围（可选）", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                data_stats["pH"]["min"] = st.number_input("pH最小值", value=float(data_stats["pH"]["min"]), format="%.2f")
                data_stats["pH"]["max"] = st.number_input("pH最大值", value=float(data_stats["pH"]["max"]), format="%.2f")
            with col2:
                data_stats["PSS"]["min"] = st.number_input("PSS最小值", value=float(data_stats["PSS"]["min"]), format="%.2f")
                data_stats["PSS"]["max"] = st.number_input("PSS最大值", value=float(data_stats["PSS"]["max"]), format="%.2f")
            with col3:
                data_stats["SOM"]["min"] = st.number_input("SOM最小值", value=float(data_stats["SOM"]["min"]), format="%.2f")
                data_stats["SOM"]["max"] = st.number_input("SOM最大值", value=float(data_stats["SOM"]["max"]), format="%.2f")
        
        if st.button("🚀 开始推导ACd标准", type="primary"):
            with st.spinner("正在生成3D网格并推导ACd标准..."):
                derive_acd_standard(model, data_stats, r2_log_scale)

if __name__ == "__main__":
    main()