# 📚 Deploying the Book Recommender System

This guide explains how to run and deploy the Streamlit application locally and to the cloud.

## Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deployment Options

### **Option 1: Streamlit Community Cloud (Easiest)**

Free hosting directly from GitHub. Best for portfolios!

#### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment app"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repo
   - Select main branch, set `app.py` as entry file
   - Click Deploy

3. **Your app is live!** Share the URL: `https://yourusername-repo-appname.streamlit.app`

**Pros:** Free, automatic updates on every push, easy sharing  
**Cons:** Limited compute, internet required

---

### **Option 2: Heroku (Classic, Paid)**

#### Steps:
1. Create `Procfile` in root:
   ```
   web: streamlit run app.py
   ```

2. Create `.streamlit/config.toml`:
   ```toml
   [server]
   headless = true
   port = $PORT
   enableCORS = false
   ```

3. Install Heroku CLI and deploy:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

**Pros:** More control, custom domain support  
**Cons:** Paid tier needed after free tier ($50+/month)

---

### **Option 3: Docker + Cloud Run / AWS / DigitalOcean**

#### Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

#### Build & Run Locally:
```bash
docker build -t recommender-app .
docker run -p 8501:8501 recommender-app
```

#### Deploy to Google Cloud Run:
```bash
gcloud run deploy --source . --platform managed
```

**Pros:** Scalable, professional infrastructure  
**Cons:** Requires setup, learning curve

---

## **Recommended: Streamlit Community Cloud** ⭐

For a portfolio project, **Streamlit Community Cloud is perfect**:

✅ Free forever (for public repos)  
✅ Automatic updates  
✅ Easy sharing  
✅ Professional URL  
✅ Perfect for showcasing projects  

### Quick Deploy (5 minutes):
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo → Deploy
4. Share the URL in your portfolio!

---

## Environment Variables (Optional)

For production deployments, create `.streamlit/secrets.toml`:

```toml
[data]
data_path = "path/to/data"
```

Access in app:
```python
import streamlit as st
secret_value = st.secrets["data"]["data_path"]
```

---

## Testing Before Deploy

```bash
# Run locally
streamlit run app.py

# Test on different machine (change to your IP):
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Visit: `http://your-ip:8501`

---

## Performance Tips

- **Cache data loading** ✓ (already done with `@st.cache_resource`)
- **Cache model training** ✓ (already done)
- **Use smaller k during development** (sidebar setting)
- **Compress data** if datasets get large

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: No module named 'streamlit'" | Run `pip install -r requirements.txt` |
| App crashes on load | Check CSV file paths are correct |
| Training takes too long | Reduce `n_steps=300` in `train_model()` or use smaller `k` |
| App is slow | Increase `patience=15` to stop training earlier |

---

## Next Steps

After deploying:

1. **Add to portfolio** → Link in GitHub profile
2. **Write blog post** → "Building & Deploying a Recommender System"
3. **Share on social media** → Impress recruiters!
4. **Enhance app** → Add comparison models, visualizations, export recommendations

---

**Happy deploying! 🚀**
