from werkzeug.security import check_password_hash

from models import user_model


def authenticate(username, password):
    user = user_model.get_user_by_username(username)
    if user and check_password_hash(user["password_hash"], password):
        user_model.update_last_login(user["id"])
        return user
    return None

