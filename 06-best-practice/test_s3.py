import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)

bucket_name = "nyc-duration"

s3.create_bucket(Bucket=bucket_name)

print("Bucket created successfully!")