import os

class S3Sync:
    def sync_folder_to_s3(self, folder: str, bucket_url: str):
        """
        Syncs a local folder to an S3 bucket.
        
        :param folder: Local folder path to sync.
        :param bucket_url: S3 bucket URL (e.g., s3://bucket-name/path).
        """
        command=f"aws s3 sync {folder} {bucket_url}"
        os.system(command)

    def sync_folder_from_s3(self, bucket_url: str, folder: str):
        """
        Syncs an S3 bucket to a local folder.
        
        :param bucket_url: S3 bucket URL (e.g., s3://bucket-name/path).
        :param folder: Local folder path to sync to.
        """
        command=f"aws s3 sync {bucket_url} {folder}"
        os.system(command)