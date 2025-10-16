"""
Simple test script for the Enhanced Scene Understanding API
"""

import requests
import json

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints"""

    print(f"Testing API at: {base_url}")
    print("-" * 40)

    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")

    print()

    # Test 2: Health check
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"✅ Health check: {response.status_code}")
        print(f"   Status: {health_data.get('status')}")
        print(f"   GPU Available: {health_data.get('gpu_available')}")
        print(f"   Models Loaded: {health_data.get('models_loaded')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

    print()

    # Test 3: System info
    try:
        response = requests.get(f"{base_url}/system-info")
        info_data = response.json()
        print(f"✅ System info: {response.status_code}")
        print(f"   CPU Count: {info_data.get('system', {}).get('cpu_count')}")
        print(f"   Memory: {info_data.get('system', {}).get('memory_total')}GB")
        print(f"   PyTorch: {info_data.get('torch', {}).get('version')}")
    except Exception as e:
        print(f"❌ System info failed: {e}")

    print("\n" + "=" * 40)
    print("API test completed!")

    # Instructions for further testing
    print("\nFor full testing:")
    print("1. Use the client_example.py script")
    print("2. Visit the interactive docs at {base_url}/docs")
    print("3. Test with actual images and videos")

if __name__ == "__main__":
    import sys

    # Allow URL to be passed as command line argument
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(url)
