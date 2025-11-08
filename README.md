# Sustainability Reports and SDGs: NLP Similarity Analysis 🌍

> MBA in Data Science Thesis - Universidade de São Paulo (USP/ICMC)

**Similarity Analysis between Sustainability Reports and the Sustainable Development Goals using Machine Learning**

## 📖 About

This research evaluates the alignment between Brazilian companies' ESG (Environmental, Social, and Governance) reports and the United Nations' 17 Sustainable Development Goals (SDGs) using Natural Language Processing and Machine Learning techniques.

As sustainability reporting becomes increasingly important for stakeholders, investors, and regulators, this work provides an automated approach to quantify how well corporate sustainability disclosures address the UN's 2030 Agenda goals.

## 🎯 Research Overview

**Challenge:** ESG reports are lengthy, unstructured documents with diverse layouts, making manual analysis time-consuming and comparison difficult.

**Solution:** Deploy and compare four different NLP/ML approaches to automatically measure semantic similarity between sustainability reports and SDG indicators:

1. **TF-IDF** (Term Frequency-Inverse Document Frequency) - Classical Bag of Words approach
2. **USE** (Universal Sentence Encoder) - Google's transfer learning model
3. **SBERT** (Bi-Encoder Sentence Transformer) - State-of-the-art sentence embeddings
4. **SimCSE** (Simple Contrastive Learning) - Advanced contrastive learning approach
   - Supervised variant
   - Unsupervised variant

**Benchmark:** Human analysis served as the gold standard for validation.

## 🔬 Methodology

### Data Sources
- **SDG Declaration:** UN's 17 Sustainable Development Goals with targets and indicators
- **Sustainability Reports:** 2021 ESG disclosures from Brazilian companies (PDF format)
- **Case Study:** Klabin S.A. - major Brazilian paper producer and exporter

### Pipeline
```
PDF Reports → Text Extraction → Data Cleaning → Text Encoding → 
Cosine Similarity Calculation → Ranking & Comparison → Human Validation
```

### Pre-processing
- PDF text extraction using PDFplumber
- Text cleaning (lowercasing, alphanumeric filtering, whitespace normalization)
- Stop-word removal and lemmatization (for TF-IDF)
- Encoding with character limit normalization

## 📊 Key Findings

The **Supervised SimCSE** model demonstrated the best performance, showing the highest congruence with human analysis. Key results:

- ✅ All methods successfully identified SDGs 12 (Responsible Consumption) and 15 (Life on Land) as most relevant
- ✅ Supervised SimCSE achieved closest alignment with expert human ranking
- ✅ Models completed analysis in 5-10 minutes vs. several hours for manual review
- ⚠️ Unsupervised models showed acceptable performance when labeled data is unavailable

## 🛠️ Technical Stack

**Languages & Frameworks:**
- Python 3.8+
- TensorFlow / TensorFlow Hub
- PyTorch

**Key Libraries:**
- `pdfplumber` - PDF text extraction
- `spacy` - NLP preprocessing
- `scikit-learn` - TF-IDF vectorization & cosine similarity
- `sentence-transformers` - SBERT and SimCSE models
- `tensorflow-hub` - Universal Sentence Encoder

**Environment:**
- Google Colab with GPU acceleration

## 💻 Usage

### Setup

```bash
# Clone the repository
git clone https://github.com/muitagabi/usp_thesis_sustainability_ml_analysis.git
cd usp_thesis_sustainability_ml_analysis

# Install dependencies
pip install pdfplumber spacy scikit-learn sentence-transformers tensorflow tensorflow-hub
python -m spacy download en_core_web_sm
```

### Running the Analysis

Open the Jupyter notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muitagabi/usp_thesis_sustainability_ml_analysis/blob/main/usp_thesis_testing_code.ipynb)

Or run locally:
```bash
jupyter notebook usp_thesis_testing_code.ipynb
```

### Code Structure

The notebook is organized in sequential sections:

1. **Installing Packages** - Dependencies setup
2. **Pre-processing** - PDF extraction and text cleaning
3. **TF-IDF** - Bag of Words similarity analysis
4. **USE** - Universal Sentence Encoder embeddings
5. **SBERT Bi-Encoder** - Sentence transformer similarity
6. **SimCSE** - Contrastive learning (supervised & unsupervised)

Each section outputs cosine similarity percentages for all 17 SDGs.

## 📂 Repository Contents

```
usp_thesis_sustainability_ml_analysis/
├── usp_thesis_testing_code.ipynb    # Main analysis code
├── usp_thesis_final_version.pdf     # Complete thesis (Portuguese)
├── sgd_declaration.pdf              # UN SDG targets & indicators
└── README.md                        # This file
```

## 🎓 Academic Context

**Institution:** Instituto de Ciências Matemáticas e de Computação (ICMC)  
**University:** Universidade de São Paulo (USP)  
**Program:** MBA in Data Science  
**Author:** Gabrielle Fidelis de Castilho  
**Advisor:** Profa. Dra. Gleici da Silva Castro Perdoná  
**Year:** 2023

## 🔑 Keywords

Machine Learning • Natural Language Processing • Deep Learning • Sustainability Reporting • ESG • Sustainable Development Goals • BERT • Transformers • Sentence Embeddings • Semantic Similarity

## 🙏 Acknowledgments

Special thanks to:
- Profa. Dra. Gleici da Silva Castro Perdoná for guidance and support
- Fundação de Apoio à Física e Química (FAFQ)
- Swedish Institute for research funding
- Daniel Thomás Ramos Franco de Sá
- Dr. Thalita Reis da Silva

## 📝 License

This project is available for academic and educational purposes.
