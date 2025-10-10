#!/usr/bin/env python3
"""
PyTorch Deep Learning Pipeline Runner
Efficient training with MinMaxScaler and one-hot encoding
"""

from pytorch_pipeline_fixed import PyTorchPipelineAdvanced
import os

def main():
    """Execute PyTorch pipeline"""
    
    data_path = '/Users/anand/Desktop/FAI/Data file/preprocessed_df.csv'
    save_path = '/Users/anand/Desktop/FAI/Deep learning'
    
    os.makedirs(save_path, exist_ok=True)
    
    print("🔥 Initializing PyTorch Deep Learning Pipeline...")
    pipeline = PyTorchPipelineAdvanced(data_path, save_path)
    
    try:
        pipeline.run_pipeline()
        print("\n✅ PyTorch pipeline completed successfully!")
        print(f"📁 Scaler saved: {save_path}/scaler.pkl")
        print(f"📊 Results saved: {save_path}/pytorch_advanced_results.csv")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
