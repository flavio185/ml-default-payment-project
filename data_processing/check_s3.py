import time

import boto3
from loguru import logger


def wait_for_s3_object(bucket, key, timeout=10):
    s3 = boto3.client("s3")
    logger.info(f"Waiting for S3 object s3://{bucket}/{key} to be available...")

    for _ in range(timeout):
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError:
            time.sleep(1)
    raise TimeoutError(f"S3 object {key} not found after {timeout} seconds.")
