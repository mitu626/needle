
curl -sf "http://localhost:8100/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "needle",
    "messages": [{"role": "user", "content": "将李白的<静夜思>改写成现代诗"}],
    "max_tokens": 500,
    "temperature": 0.6,
    "stream": false
  }' 
