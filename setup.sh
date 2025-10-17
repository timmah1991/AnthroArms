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
  xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
  libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran pkg-config

# Install Chromium for kiosk mode
if apt-cache show chromium-browser >/dev/null 2>&1; then
  apt install -y chromium-browser
elif apt-cache show chromium >/dev/null 2>&1; then
  apt install -y chromium
else
  apt install -y snapd
  snap install chromium
fi

# ----------------------------
# Install pyenv
# ----------------------------
if [ ! -d "/root/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git /root/.pyenv
fi

# Add pyenv init to profile and bashrc
for file in /root/.profile /root/.bashrc; do
  grep -qxF 'export PYENV_ROOT="$HOME/.pyenv"' "$file" || echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$file"
  grep -qxF 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' "$file" || echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$file"
  grep -qxF 'eval "$(pyenv init -)"' "$file" || echo 'eval "$(pyenv init -)"' >> "$file"
done

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ----------------------------
# Install Python 3.11.9
# ----------------------------
pyenv install -s 3.11.9
pyenv global 3.11.9

# ----------------------------
# Create Python venv and install packages
# ----------------------------
APP_DIR="/home/anthroarms/AnthroArms"
VENV_DIR="$APP_DIR/venv"

mkdir -p "$APP_DIR"
cd "$APP_DIR"

if [ ! -d "$VENV_DIR" ]; then
  /root/.pyenv/versions/3.11.9/bin/python -m venv venv
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install opencv-python mediapipe flask
deactivate

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
# Set up app.py as a service using venv
# ----------------------------
cat >/etc/systemd/system/anthroarms.service <<EOF
[Unit]
Description=AnthroArms Flask App
After=network.target

[Service]
WorkingDirectory=/home/anthroarms/AnthroArms
ExecStart=/home/anthroarms/AnthroArms/venv/bin/python /home/anthroarms/AnthroArms/app.py
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
# Configure Chromium Kiosk Mode
# ----------------------------
mkdir -p /home/pi/.config/lxsession/LXDE-pi
cat >/home/pi/.config/lxsession/LXDE-pi/autostart <<EOF
@xset s off
@xset -dpms
@xset s noblank
@chromium --noerrdialogs --disable-infobars --kiosk http://localhost:5000
EOF
chown -R pi:pi /home/pi/.config

# ----------------------------
# Auto-login and GUI boot
# ----------------------------
systemctl set-default graphical.target
systemctl enable lightdm.service

echo "✅ Setup complete! System will reboot now."
reboot