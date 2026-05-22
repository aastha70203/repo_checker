# Deployment Guide

## Streamlit Cloud

1. Push this repository to GitHub.
2. Open Streamlit Cloud and create a new app from the repository.
3. Set the entrypoint to `app.py`.
4. Add secrets:

```toml
GEMINI_API_KEY = "your_google_ai_studio_key_here"
OPENAI_API_KEY = "your_openai_key_here"
GITHUB_TOKEN = "optional_token_for_pr_comments"
```

5. Deploy and test with `karpathy/micrograd`.

## HuggingFace Spaces

1. Create a new Space with the Streamlit SDK.
2. Upload this repository.
3. Add the same secrets in Space settings.
4. Confirm that the app starts from `app.py`.

## Demo Checklist

- Run a small public repo live, such as `karpathy/micrograd`.
- Show provider selection between Google AI Studio and OpenAI.
- Show severity/category/confidence filters.
- Show the `verify this` tab for low-confidence comments.
- Download Markdown or JSON output.
- If claiming the GitHub API bonus, post comments to a real PR and capture the result.
