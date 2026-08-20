from flask import request, session

from models import log_model


def write_log(action, target_type=None, target_id=None, description=None):
    user_id = session.get("user_id")
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    log_model.create_log(user_id, action, target_type, target_id, description, ip_address)

