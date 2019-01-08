import os, zipfile

def zip_files( files, zip_name ):
    zip = zipfile.ZipFile( zip_name, 'w', zipfile.ZIP_DEFLATED )
    for file in files:
        print ('compressing', file)
        zip.write( file )
    zip.close()
    print ('compressing finished')


def make_zip(source_dir, output_filename):
	zipf = zipfile.ZipFile(output_filename, 'w')
	pre_len = len(os.path.dirname(source_dir))
	for parent, dirnames, filenames in os.walk(source_dir):
		for filename in filenames:
			pathfile = os.path.join(parent, filename)
			arcname = pathfile[pre_len:].strip(os.path.sep)
			zipf.write(pathfile, arcname)
	zipf.close()
source_dir = './results'
filename = 'res.zip'
make_zip(source_dir,filename)