#!/usr/bin/env bash
git clone https://github.com/traildb/traildb traildb-core && pushd traildb-core
wget https://mirrors.kernel.org/ubuntu/pool/universe/j/judy/libjudy-dev_1.0.5-5_amd64.deb \
     https://mirrors.kernel.org/ubuntu/pool/universe/j/judy/libjudydebian1_1.0.5-5_amd64.deb
sudo dpkg -i libjudy-dev_1.0.5-5_amd64.deb libjudydebian1_1.0.5-5_amd64.deb
sudo apt-get update
sudo apt-get install -y libjudy-dev libarchive-dev pkg-config build-essential
sudo python ./waf configure
sudo python ./waf build
sudo python ./waf install