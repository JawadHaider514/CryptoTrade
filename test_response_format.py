import sys
sys.path.insert(0, 'src')
from crypto_bot.server.advanced_web_server import app
import json

with app.test_client() as client:
    response = client.get('/api/predictions')
    data = response.get_json()
    
    # Print 2 complete signals with all fields
    if data.get('predictions'):
        for i, (symbol, pred) in enumerate(list(data['predictions'].items())[:2]):
            print(f'\n═══════════ {symbol} ═══════════')
            print(json.dumps(pred, indent=2, default=str))
    
    print(f'\n✅ Total signals: {data.get("count")}')
    print(f'✅ Response success: {data.get("success")}')
    print(f'✅ Warming up: {data.get("warming_up")}')
