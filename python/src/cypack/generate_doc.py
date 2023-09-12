#ChannelAttribution: Markov model for online multi-channel attribution
#Copyright (C) 2015 - 2023  Davide Altomare and David Loris <https://channelattribution.io>
#
#ChannelAttribution is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#ChannelAttribution is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with ChannelAttribution.  If not, see <http://www.gnu.org/licenses/>.

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
				

if not os.path.exists(os.path.join(dir,"docs/")):
    os.makedirs(os.path.join(dir,"docs/"))				

os. chdir(os.path.join(dir,"docs/"))
os.system("make clean && make html")
os.system("sphinx-build -b rinoh . _build/rinoh")
