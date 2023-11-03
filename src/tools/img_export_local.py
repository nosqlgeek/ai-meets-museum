import click
import os
import csv
from PIL import Image


'''
An inventory number is a string that has always 6 digits. 
If the numeric value is shorter than 6 digits, then we need to prepend zeros.
'''
def num_to_str(fname_prefix):
    try:
        # Check if the input is a number
        # Strings like 0_1_2 can be converted to int
        result = str(int(str(fname_prefix).replace('_','#').replace('-','#')))
        while len(result) < 6:
            result = '0' + result
        return result
    except ValueError:
        return fname_prefix


@click.command()
@click.option('-c', '--csv_file', type=str, required=True, help='The CSV file that has the MulPfadS, MulDateiS, and MulExtentS columns')
@click.option('-b', '--img_src_base', type=str, required=True, help='The base folder under which the images can be found')
def click_main(csv_file, img_src_base):

    with open(csv_file, newline='') as f:

        r = csv.DictReader(f, delimiter=';')

        for row in r:

            img_path = row['MulPfadS'].replace('\\','/').replace('X:', '')
            img_file = num_to_str(row['MulDateiS'])
            img_ext = row['MulExtentS']
            source_f_path = '{}{}/{}.{}'.format(img_src_base, img_path, img_file, img_ext)
            target_f_name_prefix = row['MulRefId']

            try:
                target_f_name = '{}.{}'.format(target_f_name_prefix, img_ext.lower())	 
                print("Uploading {} to {} ...".format(source_f_path, target_f_name))
                
                if os.path.isfile('./images/' + target_f_name):
                    print("Skipping ...")
                else:
                   thn = Image.open(source_f_path)
                   thn.thumbnail((500, 400))
                   thn.save('./images/' + target_f_name)

            except Exception as err:
                print('Could not upload file: {}'.format(str(err)))

if __name__ == '__main__':
    click_main()
