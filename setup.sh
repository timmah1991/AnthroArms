#!/usr/bin/env bash
set -e

# ----------------------------
# System Update & Upgrade
# ----------------------------
apt update -y && apt upgrade -y

# ----------------------------
# Install dependencies
# ----------------------------
apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
  xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git chromium-browser

# ----------------------------
# Install pyenv
# ----------------------------
if [ ! -d "/root/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git /root/.pyenv
fi

# Add pyenv init to /root/.profile and /root/.bashrc
grep -qxF 'export PYENV_ROOT="$HOME/.pyenv"' /root/.profile || echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.profile
grep -qxF 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' /root/.profile || echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.profile
grep -qxF 'eval "$(pyenv init -)"' /root/.profile || echo 'eval "$(pyenv init -)"' >> /root/.profile

grep -qxF 'export PYENV_ROOT="$HOME/.pyenv"' /root/.bashrc || echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.bashrc
grep -qxF 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' /root/.bashrc || echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc
grep -qxF 'eval "$(pyenv init -)"' /root/.bashrc || echo 'eval "$(pyenv init -)"' >> /root/.bashrc

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ----------------------------
# Install Python 3.11.9 via pyenv
# ----------------------------
pyenv install -s 3.11.9
pyenv global 3.11.9

# ----------------------------
# Set up node_exporter
# ----------------------------
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

# ----------------------------
# Set up app.py as a service
# ----------------------------
cat >/etc/systemd/system/anthroarms.service <<EOF
[Unit]
Description=AnthroArms Flask App
After=network.target

[Service]
WorkingDirectory=/home/anthroarms/AnthroArms
ExecStart=/root/.pyenv/versions/3.11.9/bin/python /home/anthroarms/AnthroArms/app.py
Restart=always
User=anthroarms
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable anthroarms
systemctl start anthroarms

# ----------------------------
# Configure Kiosk Mode (Chromium)
# ----------------------------
mkdir -p /home/pi/.config/lxsession/LXDE-pi
cat >/home/pi/.config/lxsession/LXDE-pi/autostart <<EOF
@xset s off
@xset -dpms
@xset s noblank
@chromium-browser --noerrdialogs --disable-infobars --kiosk http://localhost:5000
EOF
chown -R pi:pi /home/pi/.config

# ----------------------------
# Auto-login and GUI boot
# ----------------------------
systemctl set-default graphical.target
systemctl enable lightdm.service

echo "Setup complete! Rebooting..."
reboot