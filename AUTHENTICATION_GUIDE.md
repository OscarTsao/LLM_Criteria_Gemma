# HuggingFace Authentication Guide

To run training with Gemma models, you need to authenticate with HuggingFace and accept Google's terms.

---

## Quick Setup (5 minutes)

### Step 1: Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "gemma-training")
4. Select "Read" permissions
5. Copy the token (starts with `hf_`)

### Step 2: Accept Gemma Model Terms

1. Visit https://huggingface.co/google/gemma-3-4b-it
2. Click "Agree and access repository"
3. Fill out the form if prompted
4. Wait for approval (usually instant)

### Step 3: Authenticate Locally

**Option A: Interactive Login (Recommended)**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Option B: Environment Variable**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

**Option C: Add to .bashrc (Persistent)**
```bash
echo 'export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Verify Authentication

```bash
python -c "from transformers import AutoTokenizer; print('Testing...'); tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-it'); print('✓ Authentication successful!')"
```

If successful, you'll see:
```
✓ Authentication successful!
```

If failed, you'll see:
```
OSError: You are trying to access a gated repo...
```

---

## Start Training

Once authenticated:

```bash
# Quick test (30 minutes)
make train-quick

# Full 5-fold CV (2-3 hours)
make train-5fold

# View results
make show-results
```

---

## Troubleshooting

### "401 Client Error: Unauthorized"
**Cause**: Token not set or invalid
**Fix**:
1. Verify token is correct at https://huggingface.co/settings/tokens
2. Re-run authentication (Step 3)

### "Access to model is restricted"
**Cause**: Haven't accepted Gemma terms
**Fix**: Visit https://huggingface.co/google/gemma-3-4b-it and click "Agree"

### "Token has expired"
**Cause**: Token was revoked or expired
**Fix**: Create a new token (Step 1)

---

## Alternative: Test Without Authentication

If you want to test the pipeline before getting authentication:

```bash
# Use RoBERTa (no auth required, already cached)
python src/training/train_gemma_hydra.py \
    experiment=quick_test \
    model.name=roberta-base
```

This won't match Gemma's performance but verifies the pipeline works.

---

## Security Notes

- **Never commit** your HuggingFace token to git
- Keep your token private (treat it like a password)
- Revoke tokens you're not using at https://huggingface.co/settings/tokens
- Use environment variables or `huggingface-cli login` (stores securely)

---

## What Models Require Authentication?

**Require Auth (Gated)**:
- google/gemma-3-4b-it ✓ (Used in this project)
- google/gemma-3-12b-it ✓
- meta-llama/Llama-2-*
- meta-llama/Llama-3-*

**No Auth Required**:
- bert-base-uncased ✓ (cached)
- roberta-base ✓ (cached)
- distilbert-base-uncased ✓ (cached)
- microsoft/deberta-v3-base ✓ (cached)

---

## Next Steps After Authentication

1. **Verify setup**: `make quick-check`
2. **Quick test**: `make train-quick` (~30 min)
3. **Full training**: `make train-5fold` (~2-3 hours)
4. **View results**: `make show-results`

For more details, see:
- `TESTING_COMPLETE.md` - Full testing report
- `QUICK_START.md` - Getting started guide
- `RUN_5FOLD.md` - Training instructions
