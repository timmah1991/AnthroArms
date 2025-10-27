#!/bin/bash 

tar -czf /var/www/html/app.tar.gz /opt/anthroapp/
systemctl restart nginx
echo "Compression complete and nginx restarted."