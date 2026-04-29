[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/9FIzUWwz)

# MLU Final Project: Book Recommender System

**Course:** Mathematical Fundamentals for Machine Learning — City Colleges of Chicago

## Overview

Builds a book recommender system from scratch using Amazon book review data (~1,490 users, ~1,186 books, ratings 1–5). The technique is **model-based collaborative filtering** via matrix factorization.

## Approach

The rating matrix is factorized as **P = A × F**, where:
- **A** (n × k): user affinities for k latent factors
- **F** (k × m): book features for k latent factors
- **P** (n × m): predicted ratings for all user–book pairs

Parameters are learned by minimizing MSE (equivalent to maximizing Gaussian likelihood) via **gradient descent with PyTorch autograd**.

## Notebooks

| File | Description |
|------|-------------|
| `math-lab-final_project.ipynb` | Main notebook — data loading, vectorization, model training, evaluation |

## Submissions

| File | Description |
|------|-------------|
| `baseline_submission.csv` | Rank-1 baseline (ASIN average ratings) |
| `gd_submission.csv` | Gradient descent model (k=2, 500 steps) |
| `early_stopping_submission.csv` | GD with early stopping |
| `MATH_Final_Project_LB_submission.csv` | Best improved model (k=15, L2 reg, clamped [1,5]) |

## Key Concepts Covered

- Sparse matrix construction (score matrix S, mask matrix R)
- Matrix factorization and latent factors
- Maximum likelihood estimation → MSE loss
- Gradient descent with autograd
- Overfitting detection and early stopping
- L2 regularization
