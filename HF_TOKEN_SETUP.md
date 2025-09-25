# HuggingFace Token Setup Guide

## Quick Setup

1. **Get your HuggingFace token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token or copy an existing one
   - Make sure it has "Read" permissions

2. **Add token to `.env` file**:
   ```bash
   # Open the .env file
   nano .env
   
   # Replace 'your_token_here' with your actual token
   HF_TOKEN=hf_your_actual_token_here
   ```

3. **Test the setup**:
   ```bash
   python test_hf_token.py
   ```

## What This Fixes

✅ **Removes HuggingFace API rate limiting**  
✅ **Enables access to gated/private datasets**  
✅ **Improves dataset download reliability**  
✅ **Authenticates all HuggingFace API calls**

## How It Works

The system automatically:
- Loads the `HF_TOKEN` from `.env` file using `python-dotenv`
- Passes the token to all `load_dataset()` calls
- Uses the token for dataset streaming and downloads
- Handles authentication transparently

## Files Modified

- ✅ `.env` - Token configuration file
- ✅ `train.py` - Loads environment variables at startup  
- ✅ `main.py` - Loads environment variables at startup
- ✅ `core/data/data_registry.py` - Uses token in dataset loading
- ✅ `requirements.txt` - Added python-dotenv dependency

## Security

- ✅ `.env` file should be in `.gitignore` (not committed to git)
- ✅ Token is masked in logs and test output for security
- ✅ Only used for HuggingFace authentication, not stored elsewhere

## Troubleshooting

**Issue**: Still getting rate limited  
**Solution**: Make sure your token is valid and has Read permissions

**Issue**: "No HF_TOKEN found"  
**Solution**: Check that `.env` file exists and contains `HF_TOKEN=your_token`

**Issue**: "Token is placeholder value"  
**Solution**: Replace `your_token_here` with your actual HuggingFace token