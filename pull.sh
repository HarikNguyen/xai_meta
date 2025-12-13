#!/bin/bash

# Đọc thông tin từ file .env (nếu có)
if [ -f .env ]; then
  source .env
fi

# Kiểm tra xem IP và PORT có được nhập không
if [ -z "$1" ] || [ -z "$2" ]; then
  # Nếu chưa có IP và PORT, yêu cầu nhập và lưu vào file .env
  if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Vui lòng nhập IP và PORT lần đầu tiên!"
    read -p "IP: " IP
    read -p "Port: " PORT
    
    # Lưu vào .env
    echo "IP=$IP" > .env
    echo "PORT=$PORT" >> .env
  fi
else
  # Nếu IP và PORT được nhập trên dòng lệnh, lưu vào .env
  IP=$1
  PORT=$2
  echo "IP=$IP" > .env
  echo "PORT=$PORT" >> .env
fi

# Hiển thị thông tin và thực hiện rsync pull
echo "Đang kéo dữ liệu từ server $IP, Port: $PORT"

rsync -avz -e "ssh -p $PORT" root@$IP:/workspace ./
