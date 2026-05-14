# 🚀 Quick Start Guide

## Run Locally (2 minutes)

```bash
cd "/Users/ziahabibi/Desktop/Big Box/Recommender-System2"
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

## Deploy to Streamlit Cloud (5 minutes, FREE) ⭐

**Best option for your portfolio!**

### Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Add book recommender Streamlit app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/recommender-system.git
   git push -u origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo → `app.py`
   - Click "Deploy"

3. **Done!** 🎉  
   Your app is now live and shareable.

---

## Alternative: Deploy with Docker

```bash
docker build -t recommender-app .
docker run -p 8501:8501 recommender-app
```

---

## What the App Does

✨ **Interactive recommender:**
- Select any user ID
- See top-10 personalized recommendations
- View predicted ratings for unrated books
- Adjust latent factors (k) and get new recommendations
- See model performance metrics

---

## Features

🎯 **Built-in optimizations:**
- ✓ Cached model training (fast reload)
- ✓ L2 regularization (prevents overfitting)
- ✓ Early stopping (optimal model)
- ✓ Prediction clamping [1-5] (realistic ratings)
- ✓ Live training visualization

📊 **Metrics shown:**
- Train/Validation RMSE
- Model size & parameters
- Sparsity analysis
- User's previous ratings

---

## Customization

### Change hyperparameters in `app.py`:

```python
# Line ~85 - train_model() function
A, F = train_model(
    S_train, R_train, S_val, R_val,
    k=15,           # Latent factors
    lr=10.0,        # Learning rate
    n_steps=300     # Training iterations
)
```

### Adjust sidebar defaults:

```python
# Line ~110
k = st.slider("Number of Latent Factors (k)", 2, 20, 15)  # Default: 15
top_k = st.slider("Number of Recommendations", 3, 20, 10)  # Default: 10
```

---

## File Structure

```
Recommender-System2/
├── app.py                    # Streamlit app (main deployment file)
├── requirements.txt          # Python dependencies
├── DEPLOYMENT.md            # Full deployment guide
├── Procfile                 # Heroku config (optional)
├── .streamlit/
│   └── config.toml         # Streamlit theme & settings
├── math-lab-final_project.ipynb  # Original notebook
├── MATH_Final_Project_Data_training.csv
├── MATH_Final_Project_Data_test_features.csv
└── [submission CSVs...]
```

---

## Troubleshooting

**"ModuleNotFoundError: streamlit"**
```bash
pip install streamlit
```

**App is slow**
- Reduce `n_steps=300` → `n_steps=150`
- Or increase `patience=15` → `patience=10`

**CSV file not found**
- Make sure CSVs are in the same directory as `app.py`

---

## Next Steps to Impress

1. ✓ **Deploy it** (you're here!)
2. 📝 **Write blog post** explaining the math
3. 🎨 **Enhance UI** - add visualizations, comparisons
4. 🔗 **Add to portfolio** - link in GitHub profile
5. 📊 **Compare models** - add baselines, show improvements

---

**Questions?** Check `DEPLOYMENT.md` for full guide.

Good luck! 🚀
