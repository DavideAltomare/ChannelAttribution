import os

dir=os.path.dirname(__file__)

TAG = '#start py'

tag_found = False
with open(os.path.join(dir, 'ChannelAttribution.pyx')) as in_file:
    with open(os.path.join(dir,'docs/ChannelAttribution.py'), 'w') as out_file:
        for line in in_file:
            if not tag_found:
                if line.strip() == TAG:
                    tag_found = True
            else:
                out_file.write(line)
				
				

os. chdir(os.path.join(dir,"docs/"))
os.system("make clean && make html")
os.system("sphinx-build -b rinoh . _build/rinoh")