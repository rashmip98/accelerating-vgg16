PENNID='87697411'
BUCKET_NAME=ese539.${PENNID}
aws s3 mb s3://${BUCKET_NAME} --region us-east-1              # Create an S3 bucket (choose a unique bucket name)
aws s3 mb s3://${BUCKET_NAME}/dcp                             # Create folder for your tarball files
touch FILES_GO_HERE.txt                                       # Create a temp file
aws s3 cp FILES_GO_HERE.txt s3://${BUCKET_NAME}/dcp/          # Which creates the folder on S3
rm FILES_GO_HERE.txt                                          # cleanup

aws s3 mb s3://${BUCKET_NAME}/logs                            # Create a folder to keep your logs
touch LOGS_FILES_GO_HERE.txt                                  # Create a temp file
aws s3 cp LOGS_FILES_GO_HERE.txt s3://${BUCKET_NAME}/logs/    # Which creates the folder on S3
rm LOGS_FILES_GO_HERE.txt