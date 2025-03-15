# -*- coding: utf-8 -*-
"""
报警处理器模块

处理各种报警
"""

import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertHandler:
    """
    报警处理器基类

    提供报警处理的基本接口
    """

    def __init__(self, name: str):
        """
        初始化报警处理器

        参数:
        name (str): 处理器名称
        """
        self.name = name

    def handle(self, alert_data: Dict):
        """
        处理报警（子类实现）

        参数:
        alert_data (Dict): 报警数据
        """
        raise NotImplementedError("子类必须实现handle方法")


class ConsoleAlertHandler(AlertHandler):
    """
    控制台报警处理器

    将报警输出到控制台
    """

    def __init__(self, name: str = "console"):
        """
        初始化控制台报警处理器

        参数:
        name (str): 处理器名称
        """
        super().__init__(name)

    def handle(self, alert_data: Dict):
        """
        处理报警

        参数:
        alert_data (Dict): 报警数据
        """
        timestamp = alert_data.get('timestamp', datetime.now())
        monitor = alert_data.get('monitor', 'unknown')
        alert_type = alert_data.get('type', 'unknown')
        message = alert_data.get('message', 'No message')

        logger.warning(f"[{timestamp}] [{monitor}] [{alert_type}] {message}")


class FileAlertHandler(AlertHandler):
    """
    文件报警处理器

    将报警写入文件
    """

    def __init__(self, 
                 file_path: str,
                 name: str = "file"):
        """
        初始化文件报警处理器

        参数:
        file_path (str): 文件路径
        name (str): 处理器名称
        """
        super().__init__(name)
        self.file_path = file_path

        # 创建目录
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def handle(self, alert_data: Dict):
        """
        处理报警

        参数:
        alert_data (Dict): 报警数据
        """
        timestamp = alert_data.get('timestamp', datetime.now())
        monitor = alert_data.get('monitor', 'unknown')
        alert_type = alert_data.get('type', 'unknown')
        message = alert_data.get('message', 'No message')

        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] [{monitor}] [{alert_type}] {message}\n")
                f.write(json.dumps(alert_data, default=str) + "\n\n")
        except Exception as e:
            logger.error(f"写入报警文件时出错: {e}")


class EmailAlertHandler(AlertHandler):
    """
    邮件报警处理器

    通过邮件发送报警
    """

    def __init__(self, 
                 smtp_server: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 sender: str,
                 recipients: List[str],
                 name: str = "email"):
        """
        初始化邮件报警处理器

        参数:
        smtp_server (str): SMTP服务器
        smtp_port (int): SMTP端口
        username (str): 用户名
        password (str): 密码
        sender (str): 发件人
        recipients (List[str]): 收件人列表
        name (str): 处理器名称
        """
        super().__init__(name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients

    def handle(self, alert_data: Dict):
        """
        处理报警

        参数:
        alert_data (Dict): 报警数据
        """
        timestamp = alert_data.get('timestamp', datetime.now())
        monitor = alert_data.get('monitor', 'unknown')
        alert_type = alert_data.get('type', 'unknown')
        message = alert_data.get('message', 'No message')

        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = f"[报警] [{monitor}] [{alert_type}] {message}"

        # 邮件正文
        body = f"""
        时间: {timestamp}
        监控: {monitor}
        类型: {alert_type}
        消息: {message}

        详细信息:
        {json.dumps(alert_data, indent=2, default=str)}
        """

        msg.attach(MIMEText(body, 'plain'))

        try:
            # 连接SMTP服务器
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)

            # 发送邮件
            server.send_message(msg)

            # 关闭连接
            server.quit()

            logger.info(f"已发送邮件报警: {message}")
        except Exception as e:
            logger.error(f"发送邮件报警时出错: {e}")


class WebhookAlertHandler(AlertHandler):
    """
    Webhook报警处理器

    通过Webhook发送报警
    """

    def __init__(self, 
                 webhook_url: str,
                 headers: Dict = None,
                 name: str = "webhook"):
        """
        初始化Webhook报警处理器

        参数:
        webhook_url (str): Webhook URL
        headers (Dict): 请求头
        name (str): 处理器名称
        """
        super().__init__(name)
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}

    def handle(self, alert_data: Dict):
        """
        处理报警

        参数:
        alert_data (Dict): 报警数据
        """
        try:
            # 发送请求
            response = requests.post(
                self.webhook_url,
                headers=self.headers,
                json=alert_data,
                timeout=10
            )

            # 检查响应
            response.raise_for_status()

            logger.info(f"已发送Webhook报警: {alert_data.get('message', '')}")
        except Exception as e:
            logger.error(f"发送Webhook报警时出错: {e}")