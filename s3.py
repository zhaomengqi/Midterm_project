import boto3
from botocore.client import Config
from boto.s3.connection import S3Connection


ACCESS_KEY_ID = 'AKIAJANYK5DJH5HGRJSA'
ACCESS_SECRET_KEY = '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'

aws_connection = S3Connection(ACCESS_KEY_ID, ACCESS_SECRET_KEY)
bucket = aws_connection.get_bucket('zillowdata')

for file_key in bucket.list():
    print(file_key.name)
BUCKET_NAME=aws_connection.create_bucket('pleasepassokkk')

BUCKET_NAME = 'pleasepassokkk'
FILE_NAME='ok.py'

data = open(FILE_NAME, 'rb')

s3 = boto3.resource(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

s3.Bucket(BUCKET_NAME).put_object(Key=FILE_NAME, Body=data,ACL='public-read')




# Print out the bucket list

print ("Done")