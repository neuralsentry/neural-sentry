# Model Evaluation

This markdown will evaluate the following models:

- StarEncoder
- CodeBert
- CodeGen
- FLAN-T5
- CodeTrans

The architecture, dataset, and training approaches of each model are compared.

## Criterias

- Trained on C/C++
- Trained on Natural Language
  - Prefably also with Git commits
- Architecture
  - Encoder (preferred)
  - Decoder
- Learning Objective
  - Either Masked Language Modelling (MLM) or Casual Language Modelling (CLM)
  - Both can be fine-tuned for text classification

## StarEncoder

- [Blog](https://huggingface.co/blog/starcoder)
- [Paper](https://arxiv.org/pdf/2305.06161.pdf)
- [GitHub](https://github.com/bigcode-project/bigcode-encoder)
- [HuggingFace](https://huggingface.co/bigcode/starencoder)
- [Dataset](https://huggingface.co/datasets/bigcode/starcoderdata)
- [Dataset Search](https://huggingface.co/spaces/bigcode/search)
- [Dataset Portrait](https://stack.dataportraits.org/)

### Overview

- Released 2023
- Architecture
  - Encoder, Bi-directional (from Bert)
- Learning Objective
  - MLM
  - Next Sentence Prediction (NSP)
- Dataset (The Stack, Google BigQuery)
  - Natural Language
    - GitHub issues `54 GB`
    - Git commits `64 GB`
  - Code: Over 80 Languages
    - C/C++ `103 GB`
- 1 Checkpoint
  - 125M

#### Pros

- Trained on a lot of C/C++
- Trained on a lot of Git commits and GitHub issues
  - Includes linux, httpd, and openssh-portable repositories
- Is Encoder Architecture
- MLM Learning Objective
- Checkpoint is small enough to run on consumer GPUs

#### Cons

- Some model configuration required to only use the MLM objective
  - Uses the training input: `[CLS]{Snippet-1}[SEP]{Snippet-2}[SEP]`
  - Solutions:
    - Fine-tune the model
    - Or, proceed straight to text-classification

## CodeGen

- [Paper](https://arxiv.org/pdf/2203.13474.pdf)
- [GitHub](https://github.com/salesforce/CodeGen)
- [HuggingFace](https://huggingface.co/docs/transformers/model_doc/codegen)

### Overview

- Released 2022
- Architecure
  - Decoder, Autoregressive
- Learning Objective
  - Next-token Prediction (CLM)
- 3 Dataset Stages:
  1. CodeGen-NL
     - Dataset: The Pile
     - Natural Language `1159.04 GB`, `354.7B Tokens`
     - Code `95.16 GB`, `31.6B Tokens`
  2. CodeGen-Multi
     - Dataset: Google BigQuery
     - Code `340 GB`, `119.3B Tokens`
       - C/C++ `119 GB`, `19.B Tokens`
  3. CodeGen-Mono
     - Dataset: BigPython
     - Code consists of Python, not necessary for our use case
- 4 Checkpoints per variant
  - 350M, 2.7B, 6.1B, 16.1B

#### Pros

- Trained on a lot of C/C++
- Available checkpoints for small model
  - 350M, 2.7B
  - Can run on consumer GPUs

#### Cons

- Architecture not as ideal for text classification fine-tuning
- Learning objective is CLM, not MLM
  - However, there is a newer version `CodeGen2`, 2023
    - Adds MLM training objective
    - Encoder-Decoder architecture
    - Uses more languages and more data, including C and C++, using `The Stack` dataset
- Not explicitly trained on Git commits
  - However, it may have slightly learned from the commit messages in `The Pile` dataset

## CodeTrans

- [Paper](https://arxiv.org/abs/2104.02443)
- [GitHub](https://github.com/agemagician/CodeTrans)
- [HuggingFace](https://huggingface.co/models?search=code_trans)
- [Datasets](https://www.dropbox.com/sh/mzxa2dq30gnot29/AABIf7wPxH5Oe0PZHJ5jPV22a?dl=0)

### Overview

- Released 2023
- Architecture
  - Encoder-decoder (based on T5)
- Learning Objective
  - Text-to-text
- Variants
  - Function Documentation Generation (Python, Java, Go, Php, Ruby, JavaScript)
    - [CodeSearchNet Corpus Collection](https://github.com/github/CodeSearchNet) Dataset
  - Source Code Summarization (Python, SQL, C#)
    - [CODENN StackOverflow](https://github.com/sriniiyer/codenn) Dataset
  - Code Comment Generation (Java only)
    - [DeepCom](https://github.com/xing-hu/DeepCom) Dataset
  - Commit Message Generation (Java only)
    - [CommitGen](https://sjiang1.github.io/commitgen) Dataset
  - API Sequence Recommendation (Java only)
    - [Deep API Learning](https://github.com/guxd/deepAPI) Dataset
  - Programming Language and Synthesis (LISP only)
    - [AlgoLisp](https://github.com/nearai/program_synthesis/tree/master/program_synthesis/algolisp) Dataset

### Pros

- Multiple Variants Trained to Perform Specific Tasks
- Source Code Summarization Model Trained on C#
- Transformer-based Encoder-Decoder Model
  - Good for Sequence-to-Sequence Tasks (e.g. Summarization, Translation)
- Uses and Evaluates Different Training Stratagies
  - Single-Task Learning, Transfer Learning, Multi-Task Learning, Multi-Task Learning with Fine-Tuning

### Cons

- **Not Trained on C dataset**
- **Not Trained to Perform Masked Language Modeling**
  - Needed for Accurate Text Classification
- No Longer State-of-the-Art
- **Model is Largely Abandoned**
  - Incomplete README, last Git commit on 1 June 2021
- Requires more work to prepare data: text-to-text format
