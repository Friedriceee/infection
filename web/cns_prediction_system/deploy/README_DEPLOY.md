# 阿里云 ECS 部署说明

推荐部署目录：

```text
/opt/cns-cdss/
├── cns_prediction_system/
└── web_model/
```

部署流程：

```bash
sudo mkdir -p /opt/cns-cdss
sudo tar -xzf cns-cdss-deploy.tar.gz -C /opt/cns-cdss
sudo bash /opt/cns-cdss/cns_prediction_system/deploy/install_on_aliyun.sh
```

部署后服务：

```bash
sudo systemctl status cns-cdss
sudo systemctl status nginx
```

默认访问：

```text
http://116.62.66.20
```

默认账号：

```text
doctor / 123456
admin / 123456
```

阿里云安全组需要放行 80 端口。若使用 HTTPS，还需要绑定域名、申请证书，并在 Nginx 中配置 443。

