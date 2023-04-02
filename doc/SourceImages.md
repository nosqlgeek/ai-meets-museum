# Source images

The encrypted images of the artifacts are available in the following S3 bucket:

* s3://museum.s3.nosqlgeeks.com:9000

The bucket can be browsed via a web GUI here:

* http://s3.nosqlgeeks.com:9001/browser/museum

Further connection details can be found inside the [source code](./../src/tools/img_export.py) of the 'image export' tool.

> The S3 bucket is password protected. The username and password are provided separately.

# FAQ 
## What is S3?

S3 is a protocol that was originally developed by Amazon. S3 stands for Simple Storage Service. 

S3 is also the name of the distributed and highly available object store that is offered as a service by Amazon Web Services. It is quite popular and is often used for Big Data or Data Lake projects that naturally focus on batch processing of huge amounts of data.

There are quite a few OSS object stores that are S3 compatible.

## How can I access the object store programmatically?

The `minio` Python client library can access the object store. All you need are the following credentials:

* Access key
* Secret key

> The credentials are provided separately.

## My image viewer app tells me that the downloaded file is not a valid image. Why is this?

Each of the images is additionally ZIP compressed and encrypted. You will need the following credential to unzip the image:

* ZIP password

> The password is provided separately.

I recommend downloading and uncompressing the images programmatically, but if you just want to access some of them for testing purposes, then:

1. Download a file via the web GUI 
2. Rename the file to a ZIP file 
3. Extract the image
4. Enter the ZIP password
