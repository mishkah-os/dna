# 🧬 DNA: Neural Network Pattern Discovery System
## نظام اكتشاف الأنماط في الشبكات العصبية

<div dir="rtl">

# 📋 نظرة عامة على المشروع

## ما هو DNA؟

**DNA** (Discovery of Neural Architectures) هو إطار عمل بحثي متقدم لاكتشاف واستخلاص الأنماط الهندسية المخفية في الشبكات العصبية المدربة.

### الفكرة الأساسية

بدلاً من معاملة الشبكات العصبية كـ "صناديق سوداء" مليئة بأرقام عشوائية، نعاملها كـ **أنظمة هندسية** لها **بنية رياضية** قابلة للاكتشاف.

```
الطريقة التقليدية:
Neural Network = 14.5M random-ish numbers
    ↓ compress blindly (SVD)
Compressed = smaller random-ish numbers
❌ فقدان أداء، لا فهم

طريقتنا (DNA):
Neural Network = weights on a geometric manifold
    ↓ discover structure (SIREN)
DNA = mathematical function that generates weights
✅ حفاظ على الأداء، فهم عميق
```

## المكونات الرئيسية

### 1. SIREN Pattern Mining System

**نظام استخلاص الأنماط باستخدام الشبكات الموجية**

#### المبدأ
```python
# بدلاً من تخزين الأوزان:
W[layer][i][j] = 0.5234  # 14.5M numbers

# نتعلم دالة تولدها:
W(x,y,z,type) = sin(ω · DNA_network(x,y,z,type))  # 650K params
```

#### المكونات
- **SIREN Architecture**: شبكات sine-based للتعلم المستمر
- **Weight Dataset**: تحويل الأوزان إلى (coordinates → values)
- **Pattern Miner**: محرك تدريب متقدم مع PSNR metrics
- **Pattern Visualizer**: 9+ أدوات تصور لرؤية الأنماط

### 2. SVD-Based Legacy System

**النظام التقليدي للمقارنة**

#### المبدأ
```python
W = U @ diag(S) @ Vh  # تفكيك SVD خطي
```

#### الاستخدام
- مرجع للمقارنة (baseline)
- أسرع لكن أقل جودة
- لا يكتشف أنماطاً غير خطية

### 3. Advanced Visualization Suite

**أدوات تصور متقدمة**

#### الأنواع
1. **3D Manifold Visualization**: رؤية البنية الهندسية
2. **Spectral Analysis**: تحليل المحتوى الترددي (FFT)
3. **t-SNE Clustering**: اكتشاف التجمعات الطبيعية
4. **Reconstruction Quality**: تحليل شامل للدقة (9 رسوم)

### 4. Comprehensive Testing Framework

**إطار اختبار شامل**

#### التغطية
- Unit tests لجميع المكونات
- Integration tests للنظام الكامل
- Benchmark suite للأداء
- Visual regression tests

---

## 🏗️ البنية المعمارية

### نظرة عامة

```
src/dna/
├── siren.py                 # SIREN architectures (3 variants)
│   ├── SpectralDNA          # Basic sine network
│   ├── HierarchicalDNA      # Multi-scale (L/M/H freq)
│   └── AdaptiveDNA          # Location-aware frequency
│
├── weight_dataset.py        # Coordinate transformation
│   ├── WeightCoordinateMapper    # Index → coordinates
│   ├── WeightDataset             # PyTorch dataset
│   └── WeightExtractorForSIREN   # Model → dataset
│
├── pattern_miner.py         # Training engine
│   ├── PatternMiner              # Main trainer
│   ├── PSNR metrics              # Quality measurement
│   └── Checkpointing             # Model saving
│
├── pattern_visualizer.py   # Visualization suite
│   ├── 3D manifold plots         # Geometric view
│   ├── Spectral analysis         # Frequency view
│   ├── t-SNE clustering          # Similarity view
│   └── Reconstruction quality    # Error analysis
│
├── models.py                # Legacy models (SVD-based)
├── extraction.py            # Weight extraction
├── decomposition.py         # SVD decomposition
├── compressor.py            # High-level API
├── config.py                # Configuration system
├── logging_utils.py         # Logging framework
└── visualization.py         # Legacy visualizations
```

### تدفق البيانات

```
[Pretrained Model]
        ↓
    Extract Weights
        ↓
[Weight Matrices] → [Coordinates + Values]
        ↓
    Train SIREN DNA
        ↓
[Compact DNA] → Generate Weights
        ↓
    Reconstruct Model
        ↓
[Rebuilt Model] + [Visualizations]
```

---

## 📊 المواصفات التقنية

### الأداء المتوقع

| Model | Original Size | DNA Size | Compression | PSNR | R² |
|-------|--------------|----------|-------------|------|-----|
| TinyBERT (4L-312D) | 14.5M | 650K | **22x** | 35-40 dB | 0.92-0.97 |
| BERT-base | 110M | 2.5M | **44x** | 33-38 dB | 0.90-0.95 |
| RoBERTa-base | 125M | 2.8M | **45x** | 32-37 dB | 0.88-0.94 |

### المتطلبات

#### الأجهزة
- **GPU**: NVIDIA GPU مع 8GB+ VRAM (مُوصى به)
- **CPU**: يعمل لكن أبطأ بكثير
- **RAM**: 16GB+ (32GB للنماذج الكبيرة)
- **Storage**: 10GB+ لحفظ النتائج

#### البرمجيات
```bash
Python >= 3.8
PyTorch >= 2.0.0
transformers >= 4.30.0
numpy >= 1.24.0
matplotlib >= 3.7.0
scikit-learn >= 1.3.0
```

### الوقت المتوقع

| المهمة | TinyBERT | BERT-base | ملاحظات |
|--------|----------|-----------|---------|
| Weight Extraction | 1-2 min | 3-5 min | CPU |
| Dataset Creation | 2-3 min | 5-10 min | CPU |
| SIREN Training (100 epochs) | 2-3 hours | 8-12 hours | GPU |
| Reconstruction | 5-10 min | 15-30 min | GPU/CPU |
| Visualization | 10-15 min | 20-30 min | CPU |
| **Total** | **~3 hours** | **~12 hours** | GPU |

---

## 🚀 الاستخدام

### Quick Start

```bash
# 1. التثبيت
cd /home/user/dna
pip install -r requirements.txt

# 2. التشغيل الكامل
python scripts/run_pattern_mining.py

# 3. عرض النتائج
cd pattern_mining_output/
ls visualizations/
```

### الاستخدام المتقدم

```bash
# تخصيص النموذج
python scripts/run_pattern_mining.py \
    --model bert-base-uncased \
    --output-dir ./bert_mining

# تخصيص DNA
python scripts/run_pattern_mining.py \
    --dna-type hierarchical \
    --hidden-dim 512 \
    --num-layers 7

# تشغيل سريع للاختبار
python scripts/run_pattern_mining.py \
    --epochs 10 \
    --batch-size 4096
```

### الاستخدام البرمجي

```python
from dna import (
    WeightExtractorForSIREN,
    PatternMiner,
    PatternVisualizer
)
from transformers import AutoModel

# 1. تحميل النموذج
model = AutoModel.from_pretrained("bert-base-uncased")

# 2. استخراج الأوزان
extractor = WeightExtractorForSIREN(model)
dataset, metadata = extractor.extract_to_dataset()

# 3. تدريب DNA
miner = PatternMiner(dna_type='hierarchical')
history = miner.fit(dataset, epochs=100)

# 4. تصور النتائج
visualizer = PatternVisualizer()
visualizer.create_comprehensive_report(
    dataset.coords,
    dataset.values,
    reconstructed_values
)
```

---

## 📈 النتائج والتحليل

### ماذا ستكتشف؟

#### 1. الأنماط الهندسية

**Clusters (التجمعات)**
```
الأوزان ليست موزعة بشكل عشوائي، بل تتجمع حسب:
- النوع: Query/Key/Value/FFN
- الطبقة: Early layers vs Late layers
- الوظيفة: Syntax vs Semantics
```

**Manifolds (الأسطح المنحنية)**
```
الأوزان تقع على سطح منخفض الأبعاد:
- dim(effective) ≈ 10-20% of dim(ambient)
- البنية ناعمة (smooth) - تتغير تدريجياً
- قابلة للتنبؤ من الإحداثيات
```

#### 2. المحتوى الطيفي

**Low Frequencies (ترددات منخفضة)**
```
تمثل:
- الأنماط الكبيرة (global structure)
- البنية العامة للطبقة
- التوجه الإجمالي

أمثلة:
- "هذه طبقة attention"
- "الاتجاه العام موجب"
```

**High Frequencies (ترددات عالية)**
```
تمثل:
- التفاصيل الدقيقة (fine details)
- القيم المحددة
- الانتقالات الحادة

أمثلة:
- "هذا الوزن بالضبط = 0.5234"
- "قفزة من -0.3 إلى +0.8"
```

#### 3. التجمعات الطبيعية

**By Weight Type**
```
Query, Key, Value تشكل تجمعات منفصلة
    → وظائف مختلفة
    → أنماط مختلفة
```

**By Layer Depth**
```
Early layers ≠ Late layers
    → مستويات تجريد مختلفة
    → Shallow: features بسيطة
    → Deep: concepts معقدة
```

**By Frequency Content**
```
Some regions: high frequency (complex)
Other regions: low frequency (smooth)
    → Adaptive representation
```

---

## 🎯 حالات الاستخدام

### 1. Model Compression (ضغط النماذج)

**المشكلة:**
```
نريد نشر BERT على جهاز محمول
BERT-base = 440 MB
Too large!
```

**الحل:**
```python
DNA_bert = extract_pattern(BERT)  # 10 MB
# على الجهاز:
BERT_mobile = DNA_bert.generate()  # نفس الأداء
```

### 2. Model Interpretability (تفسير النماذج)

**المشكلة:**
```
"لماذا النموذج أخطأ هنا؟"
Black box - لا نعرف
```

**الحل:**
```python
patterns = visualize_dna(DNA)
# نرى: pattern #42 (syntax) dominant
#      pattern #17 (semantics) weak
# ∴ خطأ syntax، ليس semantics
```

### 3. Transfer Learning (نقل التعلم)

**المشكلة:**
```
Fine-tuning BERT على مجال جديد بطيء
```

**الحل:**
```python
DNA_general = extract(BERT_base)
DNA_medical = fine_tune(DNA_general, medical_data)
# أسرع: نتعلم الفروقات فقط، لا كل شيء
```

### 4. Architecture Search (البحث المعماري)

**المشكلة:**
```
كم layer نحتاج؟ ما هو hidden_dim المثالي؟
```

**الحل:**
```python
patterns = analyze_dna(DNA_12L)
# نجد: layers 8-12 redundant
# ∴ نستطيع تقليصها إلى 8 layers
```

### 5. Pattern Library (مكتبة أنماط)

**الرؤية:**
```python
PatternLibrary = {
    "english_syntax": pattern_42,
    "arabic_morphology": pattern_73,
    "translation": pattern_156,
    ...
}

# بناء نموذج جديد:
new_model = compose(
    PatternLibrary["english_syntax"],
    PatternLibrary["translation"],
    new_custom_pattern
)
```

---

## 🔬 المنهجية العلمية

### التقييم

#### Metrics

**Compression Quality:**
```
- Compression Ratio: original_size / dna_size
- Target: > 20x
```

**Reconstruction Quality:**
```
- PSNR: Peak Signal-to-Noise Ratio
  - > 40 dB: Excellent
  - 30-40 dB: Good
  - 20-30 dB: Fair
  - < 20 dB: Poor

- R²: Coefficient of Determination
  - > 0.95: Excellent
  - 0.90-0.95: Good
  - 0.80-0.90: Fair
  - < 0.80: Poor

- MSE: Mean Squared Error
  - < 0.001: Excellent
  - 0.001-0.01: Good
  - 0.01-0.1: Fair
  - > 0.1: Poor
```

**Pattern Discovery:**
```
- Number of Clusters: كم نمط مميز؟
- Manifold Dimension: كم بعد فعال؟
- Frequency Spectrum: أي ترددات مهيمنة؟
```

#### Validation

**Cross-Model:**
```
DNA من BERT → test على RoBERTa
DNA من RoBERTa → test على BERT
    ↓
هل الأنماط عامة؟
```

**Cross-Task:**
```
DNA trained on Language → test على Translation
    ↓
هل الأنماط قابلة للتحويل؟
```

**Ablation Studies:**
```
Remove pattern #i → measure performance drop
    ↓
ما مدى أهمية كل pattern؟
```

---

## 🎓 الأساس النظري

### Manifold Hypothesis

**الفرضية:**
> High-dimensional data lies on a low-dimensional manifold

**تطبيقنا:**
```
الأوزان تعيش في ℝ¹⁴·⁵ᴹ
لكن dim(effective) ≪ 14.5M

∴ توجد بنية هندسية مخفية
```

**الدليل:**
```python
U, S, Vh = svd(weights)
energy = cumsum(S²) / sum(S²)
# 90% energy في 10% components

∴ dim(manifold) ≈ 0.1 × dim(ambient)
```

### Implicit Neural Representations

**الفكرة:**
> Represent a signal by a neural network that maps coordinates to values

**تطبيقنا:**
```
Signal = Neural Network Weights
Coordinates = (x, y, z, type) ∈ [-1,1]⁴
Values = weight values ∈ ℝ

Network: f(coords) → value
```

**الميزة:**
```
- Continuous representation
- Infinite resolution
- Compact storage
- Differentiable
```

### Spectral Analysis

**الفكرة:**
> Every function can be decomposed into frequency components

**تطبيقنا:**
```
W(x,y) = Σ aᵢⱼ sin(ωᵢx + ϕᵢ) sin(ωⱼy + ϕⱼ)

Low freq: structure
High freq: details
```

**SIREN:**
```
sin activation → learns all frequencies
ReLU → only low frequencies (spectrum bias)

∴ SIREN better for detailed patterns
```

---

## 🛠️ التطوير والمساهمة

### Setup للتطوير

```bash
# 1. Clone
git clone https://github.com/yourusername/dna.git
cd dna

# 2. Environment
python -m venv venv
source venv/bin/activate

# 3. Install dev dependencies
pip install -r requirements-dev.txt

# 4. Install package in editable mode
pip install -e .

# 5. Run tests
pytest

# 6. Check code quality
black src/ tests/
isort src/ tests/
flake8 src/
mypy src/
```

### المساهمة

#### إضافة DNA جديد

```python
# src/dna/siren.py

class MyCustomDNA(nn.Module):
    """
    Your innovative DNA architecture
    """
    def __init__(self, ...):
        super().__init__()
        # Your implementation

    def forward(self, coords):
        # Generate weights from coordinates
        return weights
```

#### إضافة Visualization

```python
# src/dna/pattern_visualizer.py

class PatternVisualizer:
    def my_new_visualization(self, data):
        """
        Your innovative visualization
        """
        # Create plot
        plt.figure()
        # ...
        plt.savefig(self.output_dir / "my_viz.png")
```

#### إضافة Tests

```python
# tests/test_my_feature.py

def test_my_feature():
    """Test your new feature"""
    # Arrange
    dna = MyCustomDNA()

    # Act
    result = dna(coords)

    # Assert
    assert result.shape == expected_shape
```

---

## 📚 الموارد والمراجع

### الأوراق البحثية الأساسية

1. **SIREN (2020)**
   ```
   "Implicit Neural Representations with Periodic Activation Functions"
   Sitzmann et al., NeurIPS 2020
   ```

2. **Manifold Learning (2000)**
   ```
   "A Global Geometric Framework for Nonlinear Dimensionality Reduction"
   Tenenbaum et al., Science 2000
   ```

3. **Neural Compression (2023)**
   ```
   "The Lottery Ticket Hypothesis"
   Frankle & Carbin, ICLR 2019
   ```

### الدورات والتعليم

1. **Geometric Deep Learning**
   - [geometricdeeplearning.com](https://geometricdeeplearning.com)

2. **Manifold Learning**
   - Coursera: "Dimensionality Reduction"

3. **Signal Processing**
   - MIT: "Signals and Systems"

### الكود والأدوات

1. **PyTorch**
   - [pytorch.org](https://pytorch.org)

2. **Transformers**
   - [huggingface.co/transformers](https://huggingface.co/transformers)

3. **Scikit-learn**
   - [scikit-learn.org](https://scikit-learn.org)

---

## ⚖️ الترخيص والاستخدام

### MIT License

```
يُسمح باستخدام، نسخ، تعديل، دمج، نشر، توزيع، ترخيص فرعي،
و/أو بيع نسخ من البرنامج، وذلك وفقاً للشروط التالية:

- يجب تضمين إشعار حقوق النشر في جميع النسخ
- البرنامج "كما هو" بدون ضمان
```

### الاستشهاد

```bibtex
@software{dna2024,
  title = {DNA: Neural Network Pattern Discovery System},
  author = {DNA Team},
  year = {2024},
  url = {https://github.com/yourusername/dna},
  note = {SIREN-based pattern mining for neural networks}
}
```

---

## 🙏 الشكر والتقدير

### المساهمون الأساسيون

- **Architecture Design**: DNA Team
- **SIREN Implementation**: Based on Sitzmann et al.
- **Visualization Suite**: DNA Team
- **Documentation**: DNA Team

### المجتمع

شكراً لكل من ساهم بـ:
- Ideas and suggestions
- Bug reports
- Code contributions
- Documentation improvements

### الأدوات والمكتبات

- PyTorch team
- HuggingFace team
- Scikit-learn contributors
- Matplotlib developers

---

## 📞 الدعم والتواصل

### للأسئلة التقنية
- GitHub Issues: [github.com/yourusername/dna/issues](https://github.com/yourusername/dna/issues)

### للمناقشات
- GitHub Discussions: [github.com/yourusername/dna/discussions](https://github.com/yourusername/dna/discussions)

### للتحديثات
- Follow on Twitter: [@dna_project](https://twitter.com/dna_project)

---

**Built with 🧬 by the DNA Team**

**"Discovering the mathematics of intelligence, one pattern at a time."**

</div>

---

# 🌍 Project Overview (English)

## What is DNA?

**DNA** (Discovery of Neural Architectures) is an advanced research framework for discovering and extracting hidden geometric patterns in trained neural networks.

## Core Innovation

Instead of treating neural networks as "black boxes" filled with random numbers, we treat them as **geometric systems** with **mathematical structure** that can be discovered.

### The Transformation

```
Traditional: Neural Network → Blind Compression → Loss of Quality
Our Approach: Neural Network → Pattern Discovery → Mathematical Function
```

## Key Components

1. **SIREN Pattern Mining**: Sine-based networks learning continuous functions
2. **Weight Dataset**: Coordinate transformation system
3. **Pattern Miner**: Advanced training engine with PSNR metrics
4. **Visualization Suite**: 9+ tools to see the patterns

## Expected Results

| Model | Compression | PSNR | R² |
|-------|-------------|------|-----|
| TinyBERT | 22x | 35-40 dB | 0.92-0.97 |
| BERT-base | 44x | 33-38 dB | 0.90-0.95 |

## Use Cases

1. **Model Compression**: Deploy on edge devices
2. **Interpretability**: Understand why models make decisions
3. **Transfer Learning**: Fast adaptation to new domains
4. **Architecture Search**: Find optimal configurations
5. **Pattern Library**: Build models from reusable patterns

## Getting Started

```bash
pip install -r requirements.txt
python scripts/run_pattern_mining.py
```

## Learn More

- [SIREN Pattern Mining Guide](../SIREN_PATTERN_MINING.md)
- [Engineering Manifesto](ENGINEERING_MANIFESTO.md)
- [Theoretical Foundation](THEORETICAL_FOUNDATION.md)
- [API Documentation](API_DOCUMENTATION.md)

---

**"Reverse engineering the mathematics of intelligence"**

</div>
