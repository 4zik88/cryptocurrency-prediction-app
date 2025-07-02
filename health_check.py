#!/usr/bin/env python3
"""
Health Check Script for AI Crypto App
This script checks if the application is running correctly
"""

import requests
import sys
import time
import json
from datetime import datetime

def check_health():
    """Check application health"""
    try:
        # Check main application
        response = requests.get("http://localhost/_stcore/health", timeout=10)
        if response.status_code != 200:
            return False, f"App health check failed: {response.status_code}"
        
        # Check if app is responsive
        response = requests.get("http://localhost", timeout=10)
        if response.status_code != 200:
            return False, f"App not responsive: {response.status_code}"
        
        return True, "Application is healthy"
    
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to application"
    except requests.exceptions.Timeout:
        return False, "Application timeout"
    except Exception as e:
        return False, f"Health check error: {str(e)}"

def check_dependencies():
    """Check external dependencies"""
    dependencies = {
        "CryptoCompare": "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD",
        "Binance": "https://api.binance.com/api/v3/ping"
    }
    
    results = {}
    for name, url in dependencies.items():
        try:
            response = requests.get(url, timeout=5)
            results[name] = {
                "status": "ok" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results[name] = {
                "status": "error",
                "error": str(e),
                "response_time": None
            }
    
    return results

def generate_report():
    """Generate health report"""
    timestamp = datetime.now().isoformat()
    
    # Check application health
    app_healthy, app_message = check_health()
    
    # Check dependencies
    deps = check_dependencies()
    
    report = {
        "timestamp": timestamp,
        "application": {
            "status": "healthy" if app_healthy else "unhealthy",
            "message": app_message
        },
        "dependencies": deps,
        "overall_status": "healthy" if app_healthy and all(
            dep["status"] == "ok" for dep in deps.values()
        ) else "unhealthy"
    }
    
    return report

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # JSON output for monitoring systems
        report = generate_report()
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["overall_status"] == "healthy" else 1)
    
    # Human-readable output
    print("🏥 AI Crypto App Health Check")
    print("=" * 40)
    
    report = generate_report()
    
    # Application status
    app_status = report["application"]
    status_emoji = "✅" if app_status["status"] == "healthy" else "❌"
    print(f"{status_emoji} Application: {app_status['message']}")
    
    # Dependencies status
    print("\n🔗 Dependencies:")
    for name, dep in report["dependencies"].items():
        if dep["status"] == "ok":
            emoji = "✅"
            msg = f"OK ({dep['response_time']:.2f}s)" if dep['response_time'] else "OK"
        else:
            emoji = "❌"
            msg = dep.get('error', f"HTTP {dep.get('status_code', 'unknown')}")
        
        print(f"  {emoji} {name}: {msg}")
    
    # Overall status
    print(f"\n🎯 Overall Status: {report['overall_status'].upper()}")
    print(f"⏰ Checked at: {report['timestamp']}")
    
    # Exit with appropriate code
    sys.exit(0 if report["overall_status"] == "healthy" else 1)

if __name__ == "__main__":
    main() 