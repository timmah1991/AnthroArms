#!/usr/bin/env bash
sudo su 
apt update -y && apt upgrade -y
apt install python3-virtualenv
apt install curl git-core gcc make zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev
python -m venv /opt/anthroarms
git clone https://github.com/timmah1991/AnthroArms.git
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.profile
echo 'eval "$(pyenv init - bash)"' >> /root/.profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc
echo 'eval "$(pyenv init - bash)"' >> /root/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install -s 3.11.9
pyenv global 3.11.9
source "/opt/anthroarms/bin/activate" && 
pip install --upgrade pip
pip install opencv-python mediapipe flask
deactivate
cd /opt
if [ ! -d "/opt/node_exporter" ]; then
  NODE_EXPORTER_VERSION=$(curl -s https://api.github.com/repos/prometheus/node_exporter/releases/latest | grep tag_name | cut -d '"' -f 4)
  wget https://github.com/prometheus/node_exporter/releases/download/${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION#v}.linux-arm64.tar.gz -O node_exporter.tar.gz
  tar xzf node_exporter.tar.gz
  mv node_exporter-* node_exporter
  rm node_exporter.tar.gz
fi

cat >/etc/systemd/system/node_exporter.service <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
ExecStart=/opt/node_exporter/node_exporter
User=nobody
Restart=always

[Install]
WantedBy=default.target
EOF

systemctl daemon-reload
systemctl enable node_exporter
systemctl start node_exporter


cat >/etc/systemd/system/anthroarms.service <<EOF
[Unit]
Description=AnthroArms Flask App
After=network.target

[Service]
WorkingDirectory=/opt/anthroarms/AnthroArms
ExecStart=/opt/anthroarms/AnthroArms/venv/bin/python /opt/anthroarms/AnthroArms/app.py
Restart=always
User=anthroarms
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable anthroarms
systemctl start anthroarms

echo "✅ Setup complete! System will reboot now."
reboot