# Deployment Guide: Enhanced Scene Understanding API

## Overview
This guide provides step-by-step instructions for deploying the Enhanced Scene Understanding API on various free cloud platforms. Each platform has its own advantages and deployment process.

## Platform Comparison

| Platform | Free Tier | Deployment | Best For |
|----------|-----------|------------|----------|
| Railway | ✅ $5 credit/month | Git-based | Quick deployment |
| Render | ✅ 750 hours/month | Git-based | Reliable hosting |
| Hugging Face Spaces | ✅ 16GB RAM | Git-based | AI/ML projects |
| Fly.io | ✅ Limited free tier | Docker/Git | Global edge deployment |
| Vercel | ✅ Serverless | Git-based | Serverless functions |

## Prerequisites

1. **GitHub Repository**: Push your code to a GitHub repository
2. **Account Setup**: Create accounts on your chosen platform(s)
3. **Local Testing**: Ensure the API works locally

```bash
# Test locally first
uvicorn api_server:app --reload
# Visit http://localhost:8000/docs
```

## Platform-Specific Deployment

### 1. Railway Deployment (RECOMMENDED)

Railway offers the easiest deployment with generous free tier.

**Steps:**
1. Visit [Railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select your repository
4. Railway will auto-detect Python and use `railway.toml`
5. Add environment variables if needed
6. Deploy automatically starts

**Configuration Files Needed:**
- `railway.toml` ✅ (already created)
- `requirements.txt` ✅ (already created)

**Environment Variables:**
```
MODEL_CACHE_PATH=/app/models
CONFIDENCE_THRESHOLD=0.5
MAX_FRAMES_MEMORY=10
```

**Expected URL:** `https://your-project-name.up.railway.app`

### 2. Render Deployment

Render provides reliable hosting with good free tier limits.

**Steps:**
1. Visit [Render.com](https://render.com) and sign up
2. Click "New" → "Web Service"
3. Connect GitHub and select your repository
4. Configure build settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy

**Configuration Files Needed:**
- `render.yaml` ✅ (already created)
- `requirements.txt` ✅ (already created)

**Note:** Free tier spins down after 15 minutes of inactivity.

### 3. Hugging Face Spaces (BEST FOR AI/ML)

Hugging Face Spaces is perfect for AI/ML applications with generous resources.

**Steps:**
1. Visit [Huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name:** `your-username/scene-understanding-api`
   - **License:** MIT
   - **SDK:** Docker
4. Clone the repository:
```bash
git clone https://huggingface.co/spaces/your-username/scene-understanding-api
cd scene-understanding-api
```
5. Add your files and push:
```bash
# Copy all your files to the cloned directory
git add .
git commit -m "Initial deployment"
git push
```

**Configuration Files Needed:**
- `Dockerfile` ✅ (already created)
- `README.md` ✅ (already created)
- `requirements.txt` ✅ (already created)

**Expected URL:** `https://your-username-scene-understanding-api.hf.space`

### 4. Fly.io Deployment

Fly.io offers global edge deployment with generous free tier.

**Steps:**
1. Install Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```
2. Sign up and authenticate:
```bash
flyctl auth signup
# or login: flyctl auth login
```
3. Initialize your app:
```bash
flyctl launch --generate-name --no-deploy
```
4. Replace generated `fly.toml` with provided one
5. Deploy:
```bash
flyctl deploy
```

**Configuration Files Needed:**
- `fly.toml` ✅ (already created)
- `Dockerfile` ✅ (already created)

### 5. Vercel Deployment (Serverless)

Vercel is best for serverless deployment but may have limitations for heavy AI models.

**Steps:**
1. Install Vercel CLI:
```bash
npm install -g vercel
```
2. Login and deploy:
```bash
vercel login
vercel --prod
```

**Configuration Files Needed:**
- `vercel.json` ✅ (already created)

**Note:** May have cold start issues and resource limitations for large models.

## Post-Deployment Testing

After deployment, test your API:

```bash
# Health check
curl https://your-api-url/health

# Upload a test image
curl -X POST -F "file=@test_image.jpg" https://your-api-url/process-frame

# Check documentation
# Visit https://your-api-url/docs
```

## Environment Variables Configuration

Most platforms allow you to set environment variables through their web interface:

```bash
MODEL_CACHE_PATH=/app/models
CONFIDENCE_THRESHOLD=0.5
MAX_FRAMES_MEMORY=10
PORT=8000  # Usually set automatically
HOST=0.0.0.0
```

## Troubleshooting

### Common Issues:

**1. Models Not Loading:**
- Check model cache directory permissions
- Verify internet access for model downloads
- Increase memory allocation if possible

**2. Port Issues:**
- Ensure your app listens on `0.0.0.0:$PORT`
- Use environment variable `PORT` provided by platform

**3. Timeout Issues:**
- Increase startup timeout for model loading
- Use smaller models for faster initialization
- Implement health check endpoints

**4. Memory Issues:**
- Use model caching to avoid reloading
- Implement memory monitoring
- Use smaller batch sizes

### Performance Optimization:

**1. Model Selection:**
```python
# For faster deployment, use smaller models:
YOLO_MODEL = "yolov8n-seg.pt"  # Nano version
VISION_MODEL = "Salesforce/blip-image-captioning-base"  # Base version
```

**2. Caching:**
```python
# Enable model caching
os.environ["TRANSFORMERS_CACHE"] = "/app/models"
os.environ["HF_DATASETS_CACHE"] = "/app/models"
```

**3. Resource Management:**
```python
# Limit memory usage
MAX_FRAMES_MEMORY = 5  # Reduce for limited memory
BATCH_SIZE = 1  # Process one frame at a time
```

## Monitoring and Maintenance

### Health Monitoring:
- Use the `/health` endpoint for health checks
- Monitor memory usage through `/system-info`
- Set up uptime monitoring (UptimeRobot, etc.)

### Logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Model Updates:
- Models are loaded from Hugging Face Hub automatically
- Clear cache if you need to update models
- Restart the application for configuration changes

## Cost Considerations

### Free Tier Limits:
- **Railway:** $5 credit monthly (~100 hours)
- **Render:** 750 hours monthly 
- **HF Spaces:** Unlimited with 16GB RAM
- **Fly.io:** 3 shared-cpu VMs, 160GB/month
- **Vercel:** 100GB bandwidth, 100 serverless functions

### Recommendations:
1. **Start with Hugging Face Spaces** for AI/ML projects
2. **Use Railway** for quick deployment and testing
3. **Consider Render** for production-ready applications
4. **Monitor usage** to avoid unexpected charges

## Security Considerations

### Production Deployment:
1. **Environment Variables:** Never commit secrets to git
2. **CORS:** Restrict origins in production
3. **Rate Limiting:** Implement request rate limiting
4. **Input Validation:** Validate all file uploads
5. **HTTPS:** Ensure all traffic is encrypted

### Example Security Configuration:
```python
# In production, restrict CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Scaling Considerations

### Horizontal Scaling:
- Use multiple instances for high traffic
- Implement load balancing
- Consider Redis for session management

### Vertical Scaling:
- Upgrade to paid tiers for more resources
- Use GPU instances for better performance
- Implement efficient model loading

## Support and Resources

### Documentation:
- FastAPI Docs: https://fastapi.tiangolo.com/
- Platform-specific documentation linked above

### Community:
- FastAPI GitHub: https://github.com/tiangolo/fastapi
- Platform-specific communities and forums

### Monitoring:
- Set up logging and error tracking
- Use platform-provided metrics
- Consider external monitoring tools

## Next Steps

1. **Choose a Platform:** Start with Railway or Hugging Face Spaces
2. **Test Deployment:** Deploy with minimal configuration first
3. **Optimize Performance:** Tune based on usage patterns
4. **Add Features:** Implement additional endpoints as needed
5. **Monitor Usage:** Track performance and costs
6. **Scale Up:** Upgrade to paid tiers when necessary

## Summary

The Enhanced Scene Understanding API can be deployed on multiple free platforms. Hugging Face Spaces offers the best resources for AI/ML applications, while Railway provides the easiest deployment experience. Choose based on your specific needs and requirements.

Remember to:
- Start with the free tiers
- Monitor resource usage
- Optimize for your specific use case
- Plan for scaling as your application grows