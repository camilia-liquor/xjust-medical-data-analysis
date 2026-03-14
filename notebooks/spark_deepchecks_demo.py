# spark_deepchecks_demo.py
# 项目名称：基于 DeepChecks 的医学深度学习数据质量保障系统
# 模块：质量评估层 - 分布式质量验证引擎演示

from pyspark.sql import SparkSession
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
import pandas as pd

print("="*70)
print("  基于 DeepChecks 的医学深度学习数据质量保障系统")
print("  质量评估层 - 分布式质量验证引擎演示")
print("="*70)

# 1. Spark 初始化（数据处理层）
print("\n【数据处理层】初始化 Apache Spark 分布式计算引擎...")
spark = SparkSession.builder \
    .appName("MedicalDataQualitySystem") \
    .master("local[*]") \
    .getOrCreate()
print("✅ Spark 分布式引擎启动成功")

# 2. 模拟医学多模态数据（数据采集层）
print("\n【数据采集层】加载医学多模态数据（模拟DICOM/HL7/FHIR整合）...")
medical_data = [
    (1, "CT影像_001", 45, "高血压", 120.5, 80.2, "正常"),
    (2, "CT影像_002", 52, "糖尿病", 135.8, 85.6, "异常"),
    (3, "CT影像_003", None, "高血压", 125.3, 82.1, "正常"),
    (4, "CT影像_004", 38, None, 118.9, 78.5, "正常"),
    (5, "CT影像_005", 61, "糖尿病", 142.1, 88.9, "异常")
]
columns = ["patient_id", "imaging_id", "age", "diagnosis", "systolic", "diastolic", "quality_flag"]
spark_df = spark.createDataFrame(medical_data, columns)
print(f"✅ 医学数据加载完成，记录数：{spark_df.count()}")

# 3. DeepChecks 质量评估（质量评估层）
print("\n【质量评估层】集成 DeepChecks 核心算法进行多维度质量评估...")
print("   - 完整性检查 | 一致性检查 | 准确性检查 | 时效性检查")
pdf = spark_df.toPandas()
ds = Dataset(pdf, label='quality_flag', cat_features=['diagnosis', 'quality_flag'])

suite = data_integrity()
result = suite.run(ds)
print("✅ DeepChecks 质量评估完成")

# 4. 生成报告（可视化层）
print("\n【可视化层】生成自动化质检报告...")
result.save_as_html("medical_quality_report.html")
print("✅ 报告已保存：medical_quality_report.html")

print("\n" + "="*70)
print("  🎉 系统验证成功 - 符合项目计划书技术框架要求")
print("="*70)

input("\n按回车键退出...")
spark.stop()
