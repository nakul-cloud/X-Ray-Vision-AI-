import sys
print("Starting debug import...")
try:
    import xray_rag_full_pipeline
    print("Import success!")
except Exception as e:
    print(f"Import failed: {e}")
