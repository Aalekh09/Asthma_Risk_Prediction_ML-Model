#!/usr/bin/env python3
"""
Test script for optimized training to verify functionality
"""

import pandas as pd
import numpy as np
from optimized_train_model import OptimizedAsthmaPredictor

def test_optimized_training():
    """Test the optimized training with small sample"""
    print("🧪 Testing Optimized Training System...")
    
    # Initialize predictor
    predictor = OptimizedAsthmaPredictor()
    
    # Test data loading and partitioning
    try:
        data = pd.read_csv("processed-data.csv")
        print(f"✅ Successfully loaded dataset with {len(data)} rows")
        
        # Test age partitioning
        partitions = predictor.create_age_partitions(data, max_samples=100)  # Small sample for testing
        print(f"✅ Created {len(partitions)} age partitions:")
        for name, partition in partitions.items():
            print(f"   - {name}: {len(partition)} samples")
        
        # Test training on smallest partition
        if partitions:
            first_partition = list(partitions.keys())[0]
            first_data = partitions[first_partition]
            
            if len(first_data) > 50:  # Ensure we have enough data
                results, accuracy = predictor.train_partition(first_data, first_partition)
                print(f"✅ Successfully trained on {first_partition} partition")
                print(f"   Hybrid accuracy: {accuracy:.4f}")
            else:
                print(f"⚠️ Partition {first_partition} too small for training ({len(first_data)} samples)")
        
        print("\n🎉 Optimized training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_ensemble_training():
    """Test ensemble training with small sample"""
    print("\n🧪 Testing Ensemble Training...")
    
    try:
        predictor = OptimizedAsthmaPredictor()
        results, accuracy = predictor.train_ensemble_model(max_samples=500)
        print(f"✅ Ensemble training successful")
        print(f"   Hybrid accuracy: {accuracy:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING OPTIMIZED ASTHMA PREDICTION SYSTEM")
    print("=" * 60)
    
    # Run tests
    test1 = test_optimized_training()
    test2 = test_ensemble_training()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Age Partitioning Test: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Ensemble Training Test: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        print("\n📋 Next steps:")
        print("   1. Run 'python optimized_train_model.py' for full training")
        print("   2. Run 'streamlit run optimized_app.py' to start the web app")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)
