#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from crypto_bot.server.advanced_web_server import app

print("Checking registered routes...")
with app.app_context():
    routes = app.url_map.iter_rules()
    print("\nAll routes:")
    for rule in sorted(routes, key=lambda x: str(x)):
        if rule.endpoint != 'static':
            print(f"  {rule}")
    
    print("\nAPI routes only:")
    for rule in sorted(routes, key=lambda x: str(x)):
        if '/api/' in str(rule):
            print(f"  {rule}")
