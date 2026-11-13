import logging
logging.basicConfig(level=logging.INFO)



def _append_path(path, kwargs_identifier, kwargs):
    if len(kwargs) > 0:
        sub_path = ".".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"{path}/{kwargs_identifier}={sub_path}"
    else:
        return f"{path}/{kwargs_identifier}=default"