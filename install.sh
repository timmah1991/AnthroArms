#!/bin/bash

cd /opt/
curl -O http://ATBUILD.local/app.tar.gz
tar -xzf app.tar.gz
cd /opt/anthroapp/
rm -rf AnthroArms/
git clone https://github.com/timmah1991/AnthroArms.git
echo "Dependencies installed and repo updated"