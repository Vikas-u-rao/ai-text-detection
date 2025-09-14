#!/usr/bin/env python3
"""
Simple test script for the AI text detection model
"""

from transformers import pipeline
import torch

def test_model():
    """Test the AI text detection model with sample texts"""
    
    print("Testing AI Text Detection Model...")
    print("=" * 50)
    
    try:
        # Load the model
        print("Loading model: VSAsteroid/ai-text-detector-hc3")
        classifier = pipeline(
            "text-classification",
            model="VSAsteroid/ai-text-detector-hc3",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ Model loaded successfully!")
        
        # Test texts
        test_texts = [
            {
                "text": "Artificial intelligence is a rapidly evolving field that encompasses machine learning, natural language processing, and computer vision.",
                "expected": "Likely AI-generated (technical/formal)"
            },
            {
                "text": "I woke up this morning feeling refreshed after a good night's sleep. The sun was shining through my bedroom window.",
                "expected": "Likely human-written (personal/narrative)"
            },
            {
                "text": "The implementation of advanced algorithms requires careful consideration of computational complexity and optimization strategies.",
                "expected": "Could be AI-generated (technical jargon)"
            }
        ]
        
        print("\nRunning tests:")
        print("-" * 30)
        
        for i, test_case in enumerate(test_texts, 1):
            print(f"\nTest {i}:")
            print(f"Text: {test_case['text'][:100]}...")
            print(f"Expected: {test_case['expected']}")
            
            # Run inference
            results = classifier(test_case['text'])
            
            # Parse results
            for result in results[0]:
                label = result['label']
                score = result['score']
                print(f"Prediction: {label} (confidence: {score:.3f})")
            
            print("-" * 30)
        
        print("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 Model is working correctly! You can now run app.py")
    else:
        print("\n❌ There were issues with the model. Please check your installation.")