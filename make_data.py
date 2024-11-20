import time
import vertexai
import json
import pandas as pd
from vertexai.batch_prediction import BatchPredictionJob
from google.cloud import storage
import os

def upload_to_gcs(bucket_name, source_file, destination_blob):
    """Upload file to Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    return f"gs://{bucket_name}/{destination_blob}"

def create_batch_files(df, batch_size=100):
    """Split DataFrame into batches and create JSONL files"""
    os.makedirs('batch_inputs', exist_ok=True)
    
    batch_files = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_file = f'batch_inputs/batch_{i//batch_size}.jsonl'
        
        with open(batch_file, 'w') as f:
            for _, row in batch.iterrows():
                prompt = {
                    "prompt": f"make some code that functions the same as the following code: {row['output']} but is not the same. just give one example and only return the code."
                }
                f.write(json.dumps(prompt) + '\n')
        batch_files.append(batch_file)
    
    return batch_files

def process_batch(bucket_name, input_file, batch_number):
    """Process a single batch file"""
    # Upload to GCS
    gcs_input = upload_to_gcs(
        bucket_name,
        input_file,
        f"batch_inputs/batch_{batch_number}.jsonl"
    )
    
    output_prefix = f"gs://{bucket_name}/batch_outputs/batch_{batch_number}"
    
    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-1.5-flash-002",
        input_dataset=gcs_input,
        output_uri_prefix=output_prefix
    )
    
    print(f"Processing {input_file}")
    print(f"Job resource name: {batch_prediction_job.resource_name}")
    
    while not batch_prediction_job.has_ended:
        print(f"Job state: {batch_prediction_job.state.name}")
        time.sleep(10)
        batch_prediction_job.refresh()
    
    return batch_prediction_job.has_succeeded, batch_prediction_job.output_location

try:
    # Initialize vertexai
    PROJECT_ID = "your-project-id"  # Replace with your project ID
    BUCKET_NAME = "your-bucket-name"  # Replace with your bucket name
    
    vertexai.init(project=PROJECT_ID, location="us-central1")
    
    # Load your DataFrame
    df = pd.read_csv('train.csv')
    print("CSV file loaded successfully!")
    print(f"DataFrame shape: {df.shape}")
    
    # Create batch files
    batch_files = create_batch_files(df, batch_size=100)
    print(f"Created {len(batch_files)} batch files")
    
    # Process each batch
    df['generated_code'] = ''
    for i, batch_file in enumerate(batch_files):
        print(f"\nProcessing batch {i+1} of {len(batch_files)}")
        
        success, output_location = process_batch(
            input_file=batch_file,
            output_prefix=f'batch_outputs/batch_{i}'
        )
        
        if success:
            # Read the results
            with open(output_location, 'r') as f:
                results = [json.loads(line) for line in f]
            
            # Update the DataFrame
            start_idx = i * 100
            for j, result in enumerate(results):
                df.iloc[start_idx + j, df.columns.get_loc('generated_code')] = result['prediction']
            
            # Save progress
            df.to_csv('train_with_generated.csv', index=False)
            print(f"Saved progress after batch {i+1}")
        else:
            print(f"Batch {i+1} failed")
    
    print("Processing complete!")

except Exception as e:
    print(f"An error occurred: {e}")
    