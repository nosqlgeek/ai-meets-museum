import click
import csv
import tempfile
from minio import Minio
import pyminizip

TMP_PATH = tempfile.gettempdir()
S3_SERVER = 's3.nosqlgeeks.com:9000'
S3_BUCKET = 'museum'

'''
An S3 object exists if we can get the stats of that object.
'''
def s3_exists(client, bucket, name):
    try:
        client.stat_object(bucket, name)
        return True
    except Exception:
        return False



@click.command()
@click.option('-c', '--csv_file', type=str, required=True, help='The CSV file that has the MulPfadS, MulDateiS, and MulExtentS columns')
@click.option('-b', '--img_src_base', type=str, required=True, help='The base folder under which the images can be found')
@click.option('-a', '--s3_access_key', type=str, required=False, help='The S3 access key')
@click.option('-s', '--s3_secret_key', type=str, required=False, help='The S3 secret key')
@click.option('-p', '--img_encr_pwd', type=str, required=False, help='The password with which the images are encrypted before uploading them')
def click_main(csv_file, img_src_base, s3_access_key, s3_secret_key, img_encr_pwd):

    client = Minio(S3_SERVER, access_key=s3_access_key, secret_key=s3_secret_key, secure=False)

    with open(csv_file, newline='') as f:

        r = csv.DictReader(f, delimiter=';')

        for row in r:

            img_path = row['MulPfadS'].replace('\\','/').replace('X:', '')
            img_file = row['MulDateiS']
            img_ext = row['MulExtentS']
            source_f_path = '{}{}/{}.{}'.format(img_src_base, img_path, img_file, img_ext)
            target_f_name_prefix = row['MulRefId']

            try:
                data_file_path = '{}/s3_img.zip'.format(TMP_PATH)
                pyminizip.compress(source_f_path, None, data_file_path, img_encr_pwd, 5)

                # An artifact can have multiple images.
                seq = 0
                check = '{}_#.{}'.format(target_f_name_prefix, img_ext)
                while s3_exists(client, S3_BUCKET, check.replace('#', str(seq))):
                    seq = seq + 1

                target_f_name = '{}_{}.{}'.format(target_f_name_prefix, seq, img_ext)
                print("Uploading {} to {} ...".format(source_f_path, target_f_name))
                client.fput_object(S3_BUCKET, target_f_name, data_file_path)

            except Exception as err:
                print('Could not upload file: {}'.format(str(err)))

if __name__ == '__main__':
    click_main()






