#!/usr/bin/env python3
"""
Test script for the AI Text Detection model
"""

import sys
import os

# Add the current directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import AITextDetector

def test_model():
    """Test the AI text detection model with sample texts"""

    print("🧪 Testing AI Text Detection Model")
    print("=" * 50)

    # Initialize detector
    detector = AITextDetector()

    if detector.classifier is None:
        print("❌ Failed to load model")
        return False

    # Test texts
    test_cases = [
        "This is a sample human-written text about artificial intelligence and its applications in modern society.",
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing typewriters and computer keyboards.",
        "In the realm of machine learning, neural networks have revolutionized our approach to pattern recognition and data analysis.",
    ]

    print("Testing with sample texts:")
    print()

    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        result = detector.detect_text(text)
        print(f"Result: {result}")
        print()

    print("✅ Model testing completed successfully!")
    return True

if __name__ == "__main__":
    test_model()