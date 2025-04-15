# Learning from Fewer Prompts:  
Dimensionality Reduction and Optimal Coverage in LLM Prompt Embedding Space

## Overview

This project addresses a key challenge in in-context learning for Large Language Models (LLMs): **how to select a minimal yet optimal subset of prompt examples** that best inform the model’s prediction for new inputs. We approach this challenge by studying the information geometry of the model’s internal embedding space and leveraging dimensionality reduction techniques. The goal is to identify a set of prompt pairs whose embeddings are maximally diverse and span the function space governing the task.

## Motivation

While LLMs exhibit remarkable capabilities across tasks, their performance with in-context examples can be sensitive to the choice and number of examples. Rather than relying on brute-force enumeration or heuristic selection, our project aims to:
- Reduce redundancy in example selection.
- Maximize coverage of the underlying function space.
- Improve model performance by using a small, yet highly informative, set of prompt pairs.

By doing so, we contribute a principled method that connects classical statistical learning theory and modern deep learning practices.

## Problem Statement

Given a function \(f(x)\) (e.g., translating natural language queries to SQL), we ask:
- **How can we select a minimal set of examples \((x_i, y_i)\) such that the LLM accurately infers \(y_n \approx f(x_n)\) for new inputs \(x_n\)?**

This problem is recast as an issue of information geometry, where the final-token embeddings (extracted from the LLM) of prompt pairs are analyzed in a high-dimensional space before being projected to a lower-dimensional space.

## Key Concepts

### In-Context Learning & Embedding Space
- **In-Context Learning:** Leveraging a set of examples during inference so that the model can generalize to new inputs.
- **Embedding Space:** The model processes the prompt into a final-token embedding \(\mathbf{e}_i \in \mathbb{R}^d\) which encapsulates its distilled understanding.

### Dimensionality Reduction Techniques
- **PCA & SVD:** These techniques are used to reduce the dimensionality of the high-dimensional embedding vectors while preserving maximal variance.
- **Johnson–Lindenstrauss Lemma:** Guarantees that a set of high-dimensional points can be embedded into a lower-dimensional space with minimal distortion, which is critical for preserving pairwise relationships.

### Coverage and Diversity Metrics
- **Coverage:** Assessed via the proportion of variance explained by the selected subset of embeddings.
- **Diversity:** Measured through average pairwise cosine distances to ensure minimal redundancy.
- **Redundancy:** A key concern addressed by selecting examples that are approximately orthogonal in the reduced embedding space.

## Methodology and Pipeline

The project pipeline is designed as follows:

1. **Data Embedding via LLM:**  
   Natural language input and structured output pairs \((x_i, y_i)\) are embedded using a state-of-the-art LLM to obtain final-token embeddings \(\mathbf{e}_i\).

2. **Dimensionality Reduction:**  
   - Use PCA/SVD (and optionally random projections inspired by the Johnson–Lindenstrauss lemma) to project high-dimensional embeddings to a lower-dimensional space.
   - This step reveals the main axes of variation in the embedding space.

3. **Selection via Diversity and Coverage:**  
   - Develop criteria based on explained variance and pairwise cosine distance.
   - Implement a **reservoir sampling and iterative selection algorithm**:
     - Start with an initial subset of prompts.
     - Iterate through remaining embeddings, assessing whether the inclusion of a new example increases overall coverage.
     - Replace redundant examples to maintain an optimal set.

4. **Gradient-Based Search (Alternative Approach):**  
   - Interpret the reduced space as a landscape.
   - Use gradient information to identify underrepresented directions, thereby enriching the selected subset.

5. **Prompt Construction and Evaluation:**  
   - Use the selected subset to form a compact prompt.
   - Evaluate the LLM’s performance on predicting \(y_n\) for new inputs.

## Motivating Example: Natural Language to SQL

Imagine a user asking:  
> “List all customers who made a purchase in the past month.”

The LLM translates the above into an SQL query. With a dataset of 10,000 such input-output pairs, our approach selects the most representative examples to form an optimal prompt, facilitating accurate translations even when only a few examples are provided.

## Experimental Design and Results

While our initial progress has focused on establishing the theoretical framework and developing the selection algorithms, future work will detail:
- Benchmarking against traditional methods.
- Analysis of variance explained versus prompt count.
- Empirical evaluations on various tasks, starting with natural language to SQL translation.

## Broader Significance

By providing a systematic method to select high-value examples for in-context learning:
- **Efficiency Gains:** Reduce the dependency on large-scale prompt enumeration.
- **Transparency:** Offer a clear, theoretically grounded insight into prompt selection.
- **General Applicability:** The framework can extend to various domains where LLMs are applied, from natural language understanding to program synthesis.

## Getting Started

Currently, this project is in the research phase. Future releases will include:
- Detailed implementation instructions.
- Code examples and data processing pipelines.
- Experiment scripts for reproducibility.